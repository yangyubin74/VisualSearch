import numpy as np
import sqlite3
import json
import os
import sys
from tqdm import tqdm
from scipy.spatial.distance import cdist
from collections import Counter


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

# =========================================================
# 1. 지표 계산 함수 (한 쿼리에 대한 계산)
# =========================================================
def calculate_metrics_for_query(retrieved_labels, target_label, k, total_relevant_count):
    # 정답 여부 리스트 (Binary Relevance: 1 or 0)
    relevance = [1 if label == target_label else 0 for label in retrieved_labels]
    
    # 1. Precision@K
    relevant_retrieved = sum(relevance)
    precision = relevant_retrieved / k
    
    # 2. Recall@K
    # (총 정답 수가 0이면 분모 0 방지)
    recall = relevant_retrieved / total_relevant_count if total_relevant_count > 0 else 0.0
    
    # 3. Average Precision (AP)
    score = 0.0
    hits = 0.0
    for rank, rel in enumerate(relevance):
        if rel:
            hits += 1
            score += hits / (rank + 1)
            
    # AP는 검색된 정답 수 기준 (혹은 min(total, k))으로 나눔
    ap = score / min(total_relevant_count, k) if total_relevant_count > 0 else 0.0

    # 4. NDCG@K
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel:
            dcg += 1.0 / np.log2(i + 2) # log2(rank+1) -> rank는 1부터 시작하므로 i+2
            
    # IDCG (Ideal DCG) - 정답이 모두 상위에 몰려있을 때의 점수
    idcg = 0.0
    # 실제 가능한 최대 정답 수만큼 1을 채움 (최대 K개)
    num_ideal = min(total_relevant_count, k)
    for i in range(num_ideal):
        idcg += 1.0 / np.log2(i + 2)
        
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return precision, recall, ap, ndcg

# =========================================================
# 2. 전체 평가 함수
# =========================================================
def evaluate_algorithm(db_path, algo_name, k=10):
    print(f"\n>>> [{algo_name}] 평가 시작 (DB: {db_path})")
    
    if not os.path.exists(db_path):
        print(f"오류: DB 파일 없음")
        return None

    # DB 로드
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT category, feature_vector FROM features")
    rows = cursor.fetchall()
    conn.close()

    # 데이터 파싱 & 전체 정답 수 카운트
    features = []
    labels = []
    
    for cat, vec_str in rows:
        try:
            vec = np.array(json.loads(vec_str))
            features.append(vec)
            labels.append(cat)
        except:
            continue
            
    features = np.array(features)
    labels = np.array(labels)
    num_images = len(features)
    
    # 카테고리별 총 이미지 수 (Recall 분모용)
    label_counts = Counter(labels)
    
    print(f"  - 총 이미지: {num_images}장")
    
    if num_images == 0: return None

    # 거리 계산 (Euclidean)
    try:
        dists = cdist(features, features, metric='euclidean')
        np.fill_diagonal(dists, np.inf) # 자기 자신 제외
    except MemoryError:
        print("  [오류] 메모리 부족")
        return None

    # 결과 저장용 리스트
    metrics = {'P': [], 'R': [], 'mAP': [], 'NDCG': []}

    # 루프: 전체 이미지 평가
    for i in tqdm(range(num_images), desc=f"{algo_name} 분석"):
        # 정렬 후 Top-K
        sorted_indices = np.argsort(dists[i])[:k]
        retrieved_labels = labels[sorted_indices]
        target_label = labels[i]
        
        # 해당 카테고리의 전체 정답 개수 (자기 자신 1개 제외)
        total_relevant = label_counts[target_label] - 1
        
        # 지표 계산
        p, r, ap, ndcg = calculate_metrics_for_query(
            retrieved_labels, target_label, k, total_relevant
        )
        
        metrics['P'].append(p)
        metrics['R'].append(r)
        metrics['mAP'].append(ap)
        metrics['NDCG'].append(ndcg)

    # 최종 평균 계산
    final_results = {
        'Precision': np.mean(metrics['P']),
        'Recall': np.mean(metrics['R']),
        'mAP': np.mean(metrics['mAP']),
        'NDCG': np.mean(metrics['NDCG'])
    }
    return final_results

# =========================================================
# 3. 메인 실행
# =========================================================
def main():
    # 평가할 알고리즘 목록
    targets = [
        ("Hu-Moment", config.DB_PATH_HUMOMENT),
        ("Color-Moment", config.DB_PATH_COLORMOMENT),
        ("GLCM", config.DB_PATH_GLCM)
    ]
    
    print("="*80)
    print(f" [Baseline Evaluation] Traditional Algorithms Performance (Global Average @ 10)")
    print("="*80)
    print(f"{'Algorithm':<15} | {'mAP':<10} | {'NDCG':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 70)
    
    results_data = {}

    for name, path in targets:
        res = evaluate_algorithm(path, name, k=10)
        if res:
            results_data[name] = res
            print(f"\r -> {name:<15} | {res['mAP']:.4f}     | {res['NDCG']:.4f}     | {res['Precision']:.4f}     | {res['Recall']:.4f}")
        else:
            print(f" -> {name:<15} | 평가 실패 (DB 확인 필요)")

    print("="*80)
    print(" * 이 데이터를 논문의 [Table 1]에 사용하세요.")

if __name__ == "__main__":
    main()