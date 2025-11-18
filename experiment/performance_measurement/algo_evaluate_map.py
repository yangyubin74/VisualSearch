import numpy as np
import sqlite3
import json
import os
import sys
from tqdm import tqdm
from scipy.spatial.distance import cdist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import config

# =========================================================
# 1. 평가 메트릭 계산 함수들
# =========================================================

def calculate_precision_recall(is_relevant, k, total_relevant):
    """
    Precision@K와 Recall@K 계산
    
    Args:
        is_relevant: boolean array (검색된 결과의 관련성)
        k: Top-K
        total_relevant: 전체 DB에서 관련 문서의 총 개수 (자기 자신 제외)
    
    Returns:
        precision, recall
    """
    relevant_retrieved = np.sum(is_relevant[:k])
    precision = relevant_retrieved / k if k > 0 else 0.0
    
    # Recall: Top-K에서 찾은 관련 문서 수 / 전체 관련 문서 수
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    return precision, recall

def calculate_ap(is_relevant, k):
    """
    Average Precision@K 계산
    
    Args:
        is_relevant: boolean array
        k: Top-K
    
    Returns:
        AP score
    """
    score = 0.0
    relevant_cnt = 0
    
    for rank, rel in enumerate(is_relevant[:k]):
        if rel:
            relevant_cnt += 1
            score += relevant_cnt / (rank + 1)
    
    if relevant_cnt > 0:
        return score / min(relevant_cnt, k)
    return 0.0

def calculate_ndcg(is_relevant, k):
    """
    NDCG@K (Normalized Discounted Cumulative Gain) 계산
    
    Args:
        is_relevant: boolean array
        k: Top-K
    
    Returns:
        NDCG score
    """
    # DCG 계산
    dcg = 0.0
    for i, rel in enumerate(is_relevant[:k]):
        if rel:
            # rel은 0 또는 1이므로, 관련 있으면 1
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 0
    
    # IDCG 계산 (이상적인 순서: 모든 관련 문서가 앞에)
    ideal_relevance = np.sort(is_relevant)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        if rel:
            idcg += 1.0 / np.log2(i + 2)
    
    # NDCG
    if idcg > 0:
        return dcg / idcg
    return 0.0

# =========================================================
# 2. 전체 메트릭 계산 함수
# =========================================================
def calculate_all_metrics(db_path, algo_name, k=10, distance_metric='euclidean'):
    """
    mAP@K, NDCG@K, Precision@K, Recall@K를 모두 계산
    
    Args:
        db_path: 데이터베이스 경로
        algo_name: 알고리즘 이름
        k: Top-K
        distance_metric: 거리 메트릭 ('euclidean' 또는 'cityblock'=Manhattan)
    
    Returns:
        dict: {'mAP', 'NDCG', 'Precision', 'Recall'}
    """
    print(f"\n>>> [{algo_name}] 데이터베이스 로드 중... ({db_path})")
    
    if not os.path.exists(db_path):
        print(f"오류: DB 파일이 없습니다! -> {db_path}")
        return {'mAP': 0.0, 'NDCG': 0.0, 'Precision': 0.0, 'Recall': 0.0}

    # DB에서 특징 벡터와 라벨(카테고리) 로드
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, category, feature_vector FROM features")
    rows = cursor.fetchall()
    conn.close()

    # 데이터 파싱
    features = []
    labels = []
    
    for _, category, vec_str in rows:
        try:
            vec = np.array(json.loads(vec_str))
            features.append(vec)
            labels.append(category)
        except Exception as e:
            print(f"파싱 오류 발생 (건너뜀): {e}")
            continue
    
    # numpy 배열로 변환
    features = np.array(features)
    labels = np.array(labels)
    
    num_images = len(features)
    print(f"  - 총 이미지 수: {num_images}장")
    print(f"  - 특징 벡터 차원: {features.shape[1]}")
    
    if num_images == 0:
        return {'mAP': 0.0, 'NDCG': 0.0, 'Precision': 0.0, 'Recall': 0.0}

    # 전체 거리 행렬 계산
    print(f"  - 전체 거리 행렬 계산 중 (Metric: {distance_metric.upper()})...")
    try:
        dists = cdist(features, features, metric=distance_metric)
    except MemoryError:
        print("  [오류] 메모리 부족! 배치 처리가 필요합니다.")
        return {'mAP': 0.0, 'NDCG': 0.0, 'Precision': 0.0, 'Recall': 0.0}

    # 자기 자신과의 거리는 무한대로 설정
    np.fill_diagonal(dists, np.inf)
    
    # 각 카테고리별 이미지 수 미리 계산 (Recall 계산용)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels, label_counts))
    
    # 메트릭 저장 리스트
    ap_scores = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    
    print(f"  - 모든 메트릭 계산 중 (mAP@{k}, NDCG@{k}, Precision@{k}, Recall@{k})...")
    
    # 각 이미지를 쿼리로 사용
    for i in tqdm(range(num_images), desc=f"{algo_name} 평가"):
        # 거리 기준 정렬하여 상위 K개 인덱스
        sorted_indices = np.argsort(dists[i])[:k]
        
        # 검색된 이미지들의 라벨
        retrieved_labels = labels[sorted_indices]
        target_label = labels[i]
        
        # 정답 여부
        is_relevant = (retrieved_labels == target_label)
        
        # 전체 관련 문서 수 (같은 카테고리의 이미지 수 - 자기 자신 제외)
        total_relevant = label_count_dict[target_label] - 1
        
        # 1) Average Precision
        ap_scores.append(calculate_ap(is_relevant, k))
        
        # 2) NDCG
        ndcg_scores.append(calculate_ndcg(is_relevant, k))
        
        # 3) Precision@K
        relevant_retrieved = np.sum(is_relevant)
        precision = relevant_retrieved / k if k > 0 else 0.0
        precision_scores.append(precision)
        
        # 4) Recall@K
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        recall_scores.append(recall)
    
    # 평균 계산
    results = {
        'mAP': np.mean(ap_scores),
        'NDCG': np.mean(ndcg_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores)
    }
    
    return results

# =========================================================
# 3. 메인 실행 함수
# =========================================================
def main():
    print("="*80)
    print(" [Traditional Algorithms] 전체 데이터셋 성능 평가")
    print(" Metrics: mAP@10, NDCG@10, Precision@10, Recall@10")
    print(" Distance Metrics: Euclidean, Manhattan")
    print("="*80)
    
    # 두 가지 거리 메트릭으로 평가
    distance_metrics = ['euclidean', 'cityblock']  # cityblock = Manhattan
    metric_names = ['Euclidean', 'Manhattan']
    
    for dist_metric, metric_name in zip(distance_metrics, metric_names):
        print(f"\n{'#'*80}")
        print(f"# Distance Metric: {metric_name}")
        print(f"{'#'*80}")
        
        all_results = {}
        
        # 1. Color-Moment 평가
        results_color = calculate_all_metrics(
            config.DB_PATH_COLORMOMENT, "Color-Moment", k=10, distance_metric=dist_metric
        )
        all_results['Color-Moment'] = results_color
        
        # 2. GLCM 평가
        results_glcm = calculate_all_metrics(
            config.DB_PATH_GLCM, "GLCM", k=10, distance_metric=dist_metric
        )
        all_results['GLCM'] = results_glcm
        
        # 3. Hu-Moment 평가
        results_hu = calculate_all_metrics(
            config.DB_PATH_HUMOMENT, "Hu-Moment", k=10, distance_metric=dist_metric
        )
        all_results['Hu-Moment'] = results_hu
        
        # ---------------------------------------------------------
        # 결과 출력
        # ---------------------------------------------------------
        print("\n" + "="*80)
        print(f"{'Algorithm':<20} | {'mAP@10':<10} | {'NDCG@10':<10} | {'Precision@10':<13} | {'Recall@10':<10}")
        print("-" * 80)
        
        for algo, metrics in all_results.items():
            print(f"{algo:<20} | {metrics['mAP']:.4f}     | {metrics['NDCG']:.4f}     | "
                  f"{metrics['Precision']:.4f}        | {metrics['Recall']:.4f}")
        
        print("="*80)
        
        # 백분율 표시
        print("\n" + "="*80)
        print(f"{'Algorithm':<20} | {'mAP (%)':<10} | {'NDCG (%)':<10} | {'Precision (%)':<15} | {'Recall (%)':<10}")
        print("-" * 80)
        
        for algo, metrics in all_results.items():
            print(f"{algo:<20} | {metrics['mAP']*100:>6.2f}%    | {metrics['NDCG']*100:>6.2f}%    | "
                  f"{metrics['Precision']*100:>10.2f}%     | {metrics['Recall']*100:>6.2f}%")
        
        print("="*80)
    
    print("\n" + "="*80)
    print(" * 이 표의 값을 논문 [Baseline] 성능표에 사용하세요.")
    print(" * Euclidean과 Manhattan 결과를 비교하여 최적의 거리 메트릭을 선택하세요.")
    print("="*80)

if __name__ == "__main__":
    main()