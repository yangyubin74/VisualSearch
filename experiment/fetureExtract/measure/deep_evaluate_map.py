import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import json
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config 
import common_config as cfg
from common_utility import measure_process_time as process_time 

# --- 설정값 ---
TOP_K = config.TOP_K
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP
MODEL_NAME = None # [수정] 모델 이름을 저장할 전역 변수

# --- (1) [수정] 모델 이름을 인자로 받도록 수정 ---
def load_search_database(model_name: str, split='train'):
    """
    지정된 모델과 스플릿의 특징점과 파일명 로드
    """
    
    # [수정] "mobilenetv3" 대신 model_name 변수 사용
    feature_dir = Path(config.FEATURE_SAVE_DIR) / model_name / cfg.SEED_DIR
    
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

# --- (2) (수정 없음) ---
def get_category_from_path(path_str: str) -> str:
    try:
        return Path(path_str).parent.name
    except Exception:
        return "unknown"

# --- (3) (수정 없음) ---
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

# --- (4) (수정 없음) ---
def calculate_metrics(
    results: List[Dict], 
    relevant_category: str, 
    total_relevant_count: int, 
    k: int
) -> Dict:
    r = [1 if item['category'] == relevant_category else 0 for item in results]
    k_actual = len(r)
    if k_actual == 0:
       return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'map_at_k': 0.0, 'ndcg_at_k': 0.0}
    precision_at_k = np.sum(r) / k_actual
    recall_at_k = np.sum(r) / total_relevant_count if total_relevant_count > 0 else 0.0
    precisions = []
    relevant_count = 0
    for i in range(k_actual):
        if r[i] == 1:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    map_at_k = np.mean(precisions) if precisions else 0.0
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

# --- (5) [수정] main 함수 수정 ---
def main():
    
    print("\n" + "="*70)
    print(f"모델 [{MODEL_NAME}] mAP@{TOP_K} 전체 테스트 시작") # [수정]
    print(f"SEED: {cfg.SEED_DIR}")
    print("="*70)

    # 1. DB 로드 ('train' 세트)
    db_features, db_filenames = load_search_database(MODEL_NAME, split='train') # [수정]
    
    # 2. 쿼리 로드 ('test' 세트)
    query_features, query_filenames = load_search_database(MODEL_NAME, split='test') # [수정]
    
    total_queries = len(query_filenames)
    if total_queries == 0:
        print("❌ 오류: 'test' 세트에 쿼리할 이미지가 없습니다.")
        return

    # 3. 모든 AP 점수를 저장할 리스트
    all_ap_scores_euclidean = []
    all_ap_scores_manhattan = []
    
    # 4. 'test' 세트의 모든 이미지를 쿼리로 사용하여 반복
    print(f"\n[Step 3] {total_queries}개의 'test' 이미지로 mAP 계산 중...")
    
    for i in tqdm(range(total_queries), desc=f"[{MODEL_NAME}] Test Set 쿼리 처리 중"): # [수정]
        query_feat = query_features[i]
        query_path = query_filenames[i]
        relevant_category = get_category_from_path(query_path)
        
        if relevant_category not in TOTAL_RELEVANT_COUNT_MAP:
            continue
        total_relevant_count = TOTAL_RELEVANT_COUNT_MAP[relevant_category]

        # 5. 검색 및 AP(map_at_k) 계산
        eu_results = find_similar_images(query_feat, db_features, db_filenames, 'euclidean', TOP_K)
        eu_metrics = calculate_metrics(eu_results, relevant_category, total_relevant_count, TOP_K)
        all_ap_scores_euclidean.append(eu_metrics['map_at_k'])
        
        mh_results = find_similar_images(query_feat, db_features, db_filenames, 'manhattan', TOP_K)
        mh_metrics = calculate_metrics(mh_results, relevant_category, total_relevant_count, TOP_K)
        all_ap_scores_manhattan.append(mh_metrics['map_at_k'])

    # 6. 최종 mAP (Mean Average Precision) 계산
    if not all_ap_scores_euclidean:
        print("❌ 오류: 유효한 AP 점수가 없습니다.")
        return

    final_map_euclidean = np.mean(all_ap_scores_euclidean)
    final_map_manhattan = np.mean(all_ap_scores_manhattan)
    
    print("\n" + "="*70)
    print(f" 모델 [{MODEL_NAME}] - 최종 성능 평가 결과 (mAP@{TOP_K})") # [수정]
    print("="*70)
    print(f"  총 쿼리 이미지 수 (Test Set): {len(all_ap_scores_euclidean)}개")
    print(f"  - Euclidean mAP:   {final_map_euclidean:.6f}")
    print(f"  - Manhattan mAP:   {final_map_manhattan:.6f}")
    print("="*70)

if __name__ == "__main__":
    try:
        # [수정] 실행 시 모델 이름을 받도록 변경
        if len(sys.argv) == 2:
            MODEL_NAME = sys.argv[1]
            # 유효한 모델 이름인지 확인 (config.py/common_config.py 기준)
            valid_models = ["efficientnet", "autoencoder", "siamesenetwork", "mobilenetv3"]
            if MODEL_NAME not in valid_models:
                 print(f"❌ 오류: '{MODEL_NAME}'은(는) 유효한 모델 이름이 아닙니다.")
                 print(f"  사용 가능: {valid_models}")
                 sys.exit(1)
        else:
            print("\n========> 사용법: python evaluate_map_universal.py [model_name]")
            print("                (예: python evaluate_map_universal.py mobilenetv3)")
            print("                (예: python evaluate_map_universal.py efficientnet)")
            sys.exit(1)

        process_time(main)
        print("\n mAP Evaluation completed successfully!")

    except Exception as e:
        print(f"\n 스크립트 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()