import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import json

from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from common_utility import measure_process_time as process_time 

# --- 설정값 ---
TOP_K = config.TOP_K
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP
MODEL_NAME = None 

# --- (1) 모델 및 데이터 로드 ---
def load_search_database(model_name: str, split='train'):
    """
    지정된 모델과 스플릿의 특징점과 파일명 로드
    """
    feature_dir = Path(config.FEATURE_SAVE_DIR) / model_name / config.SEED_DIR
    
    all_db_features = []
    all_db_filenames = []
    
    print(f"📂 '{model_name}' - '{split}' 데이터베이스 로드 중...")
    
    if not hasattr(config, 'CLASSES'):
        sys.exit(1)

    for class_name in config.CLASSES:
        npy_path = feature_dir / f"{split}_{class_name}_features.npy"
        json_path = feature_dir / f"{split}_{class_name}_features.json"

        if not npy_path.exists() or not json_path.exists():
            continue

        try:
            db_features = np.load(npy_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                db_filenames = json.load(f)
            
            all_db_features.append(db_features)
            all_db_filenames.extend(db_filenames)
        except Exception as e:
            print(f"오류: {class_name} 데이터베이스 로드 실패: {e}")
            sys.exit(1)

    if not all_db_filenames:
        print(f"  ❌ 오류: '{split}' 스플릿에 대한 데이터베이스 파일이 전혀 없습니다.")
        print(f"  경로: {feature_dir}")
        print(f"  특징점 추출 스크립트(예: 04b_*.py)를 먼저 실행하세요.")
        sys.exit(1)

    final_db_features = np.vstack(all_db_features)
    
    print(f"  총 특징 벡터: {final_db_features.shape}")
    print(f"  총 파일명: {len(all_db_filenames)}개")
        
    return final_db_features, all_db_filenames

# --- (2) 카테고리 추출 ---
def get_category_from_path(path_str: str) -> str:
    try:
        return Path(path_str).parent.name
    except Exception:
        return "unknown"

# --- (3) 유사도 검색 ---
def find_similar_images(
    target_feature: np.ndarray,
    db_features: np.ndarray,
    db_filenames: List[str],
    metric: str = 'euclidean',
    top_k: int = TOP_K
) -> List[Dict]:
    query_features_2d = target_feature.reshape(1, -1)
    distances = pairwise_distances(
        query_features_2d, db_features, metric=metric
    ).flatten()
    top_indices = np.argsort(distances)[:top_k]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        file_path = db_filenames[idx]
        results.append({
            "rank": rank,
            "image_path": file_path,
            "category": get_category_from_path(file_path),
            "distance": float(distances[idx])
        })
    return results

# --- (4) 지표 계산 (Precision, Recall, mAP, NDCG) ---
def calculate_metrics(
    results: List[Dict], 
    relevant_category: str, 
    total_relevant_count: int, 
    k: int
) -> Dict:
    # 정답 여부 리스트 (1 or 0)
    r = [1 if item['category'] == relevant_category else 0 for item in results]
    k_actual = len(r)
    
    if k_actual == 0:
       return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'map_at_k': 0.0, 'ndcg_at_k': 0.0}
    
    # 1. Precision@K
    precision_at_k = np.sum(r) / k_actual
    
    # 2. Recall@K
    recall_at_k = np.sum(r) / total_relevant_count if total_relevant_count > 0 else 0.0
    
    # 3. mAP@K (Average Precision)
    precisions = []
    relevant_count = 0
    for i in range(k_actual):
        if r[i] == 1:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
            
    # AP는 검색된 정답 수 기준 평균 (또는 min(total, k))
    map_at_k = np.mean(precisions) if precisions else 0.0
    
    # 4. NDCG@K
    dcg = np.sum([r[i] / np.log2(i + 2) for i in range(k_actual)])
    ideal_r = sorted(r, reverse=True)
    idcg = np.sum([ideal_r[i] / np.log2(i + 2) for i in range(k_actual)])
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "map_at_k": map_at_k,
        "ndcg_at_k": ndcg_at_k
    }

# --- (5) 메인 함수 ---
def main():
    
    print("\n" + "="*80)
    print(f" [{MODEL_NAME}] Deep Learning Model Evaluation (Global Average @ {TOP_K})")
    print(f" SEED: {config.SEED_DIR}")
    print("="*80)

    # 1. DB 로드 ('train' 세트)
    db_features, db_filenames = load_search_database(MODEL_NAME, split='train')
    
    # 2. 쿼리 로드 ('test' 세트)
    query_features, query_filenames = load_search_database(MODEL_NAME, split='test')
    
    total_queries = len(query_filenames)
    if total_queries == 0:
        print("❌ 오류: 'test' 세트에 쿼리할 이미지가 없습니다.")
        return

    # 3. 지표 저장용 딕셔너리 초기화
    metrics_euclidean = {'mAP': [], 'NDCG': [], 'Precision': [], 'Recall': []}
    metrics_manhattan = {'mAP': [], 'NDCG': [], 'Precision': [], 'Recall': []}
    
    # 4. 전체 쿼리 평가
    print(f"\n[Step 3] {total_queries}개의 Test 이미지로 전체 성능 평가 중...")
    
    for i in tqdm(range(total_queries), desc=f"[{MODEL_NAME}] 평가 진행"):
        query_feat = query_features[i]
        query_path = query_filenames[i]
        relevant_category = get_category_from_path(query_path)
        
        if relevant_category not in TOTAL_RELEVANT_COUNT_MAP:
            continue
        total_relevant_count = TOTAL_RELEVANT_COUNT_MAP[relevant_category]

        # --- Euclidean 평가 ---
        eu_results = find_similar_images(query_feat, db_features, db_filenames, 'euclidean', TOP_K)
        eu_metrics = calculate_metrics(eu_results, relevant_category, total_relevant_count, TOP_K)
        
        metrics_euclidean['mAP'].append(eu_metrics['map_at_k'])
        metrics_euclidean['NDCG'].append(eu_metrics['ndcg_at_k'])
        metrics_euclidean['Precision'].append(eu_metrics['precision_at_k'])
        metrics_euclidean['Recall'].append(eu_metrics['recall_at_k'])
        
        # --- Manhattan 평가 ---
        mh_results = find_similar_images(query_feat, db_features, db_filenames, 'manhattan', TOP_K)
        mh_metrics = calculate_metrics(mh_results, relevant_category, total_relevant_count, TOP_K)
        
        metrics_manhattan['mAP'].append(mh_metrics['map_at_k'])
        metrics_manhattan['NDCG'].append(mh_metrics['ndcg_at_k'])
        metrics_manhattan['Precision'].append(mh_metrics['precision_at_k'])
        metrics_manhattan['Recall'].append(mh_metrics['recall_at_k'])

    # 5. 최종 평균 계산 및 출력
    if not metrics_euclidean['mAP']:
        print("❌ 오류: 유효한 결과가 없습니다.")
        return

    def print_row(metric_name, eu_list, mh_list):
        eu_mean = np.mean(eu_list)
        mh_mean = np.mean(mh_list)
        print(f"  {metric_name:<12} | {eu_mean:.6f}      | {mh_mean:.6f}")

    print("\n" + "="*60)
    print(f" 모델 [{MODEL_NAME}] 최종 성능표")
    print("-" * 60)
    print(f"  {'Metric':<12} | {'Euclidean':<12} | {'Manhattan':<12}")
    print("-" * 60)
    
    print_row("mAP@10", metrics_euclidean['mAP'], metrics_manhattan['mAP'])
    print_row("NDCG@10", metrics_euclidean['NDCG'], metrics_manhattan['NDCG'])
    print_row("Precision@10", metrics_euclidean['Precision'], metrics_manhattan['Precision'])
    print_row("Recall@10", metrics_euclidean['Recall'], metrics_manhattan['Recall'])
    
    print("="*60)
    print(" * 이 데이터를 논문 [Deep Learning Model] 성능 비교표에 사용하세요.")

if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            MODEL_NAME = sys.argv[1]
            valid_models = ["efficientnet", "autoencoder", "siamesenetwork", "mobilenetv3"]
            if MODEL_NAME not in valid_models:
                 print(f"❌ 오류: '{MODEL_NAME}'은(는) 유효한 모델 이름이 아닙니다.")
                 print(f"  사용 가능: {valid_models}")
                 sys.exit(1)
        else:
            print("\n========> 사용법: python deep_evaluate_map.py [model_name]")
            print("                (예: python deep_evaluate_map.py efficientnet)")
            sys.exit(1)

        process_time(main)

    except Exception as e:
        print(f"\n 스크립트 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()