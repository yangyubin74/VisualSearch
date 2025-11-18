import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import json

from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

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
        print("❌ 오류: config.CLASSES가 정의되지 않았습니다.")
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

# --- (3) 카테고리별 관련 문서 수 계산 ---
def build_category_counts(filenames: List[str]) -> Dict[str, int]:
    """
    각 카테고리별 이미지 수를 계산
    """
    category_counts = {}
    for filename in filenames:
        category = get_category_from_path(filename)
        category_counts[category] = category_counts.get(category, 0) + 1
    return category_counts

# --- (4) 지표 계산 (Precision, Recall, mAP, NDCG) ---
def calculate_metrics(
    is_relevant: np.ndarray,
    total_relevant_count: int, 
    k: int
) -> Dict:
    """
    주어진 relevance 배열로부터 모든 메트릭 계산
    
    Args:
        is_relevant: boolean array (검색된 결과의 관련성)
        total_relevant_count: 전체 DB에서 해당 카테고리의 총 문서 수
        k: Top-K
    
    Returns:
        dict with precision, recall, mAP, NDCG
    """
    k_actual = min(len(is_relevant), k)
    
    if k_actual == 0:
        return {'precision': 0.0, 'recall': 0.0, 'mAP': 0.0, 'NDCG': 0.0}
    
    r = is_relevant[:k_actual]
    relevant_retrieved = np.sum(r)
    
    # 1. Precision@K
    precision = relevant_retrieved / k_actual
    
    # 2. Recall@K
    recall = relevant_retrieved / total_relevant_count if total_relevant_count > 0 else 0.0
    
    # 3. mAP@K (Average Precision) - Traditional 코드와 동일한 방식
    score = 0.0
    relevant_count = 0
    for i in range(k_actual):
        if r[i]:
            relevant_count += 1
            score += relevant_count / (i + 1)
    
    # AP 계산: Traditional 코드와 동일하게
    if relevant_count > 0:
        map_score = score / min(relevant_count, k_actual)
    else:
        map_score = 0.0
    
    # 4. NDCG@K
    dcg = np.sum([r[i] / np.log2(i + 2) for i in range(k_actual)])
    ideal_r = np.sort(r)[::-1]
    idcg = np.sum([ideal_r[i] / np.log2(i + 2) for i in range(k_actual)])
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "mAP": map_score,
        "NDCG": ndcg
    }

# --- (5) 배치 평가 함수 (최적화된 버전) ---
def evaluate_batch(
    query_features: np.ndarray,
    query_filenames: List[str],
    db_features: np.ndarray,
    db_filenames: List[str],
    db_category_counts: Dict[str, int],
    metric: str,
    k: int = TOP_K
) -> Dict[str, List[float]]:
    """
    전체 쿼리를 배치로 평가 (고속 버전)
    
    Returns:
        dict with lists of mAP, NDCG, Precision, Recall scores
    """
    # 1. 전체 거리 행렬을 한 번에 계산 (핵심 최적화!)
    print(f"  - 전체 거리 행렬 계산 중 (Metric: {metric.upper()})...")
    distances = pairwise_distances(query_features, db_features, metric=metric)
    
    # 2. DB 카테고리 배열 미리 생성
    db_categories = np.array([get_category_from_path(f) for f in db_filenames])
    
    # 3. 각 쿼리에 대해 평가
    results = {'mAP': [], 'NDCG': [], 'Precision': [], 'Recall': []}
    
    skipped_count = 0
    
    for i in tqdm(range(len(query_filenames)), desc=f"  {metric.capitalize()} 평가"):
        query_path = query_filenames[i]
        query_category = get_category_from_path(query_path)
        
        # 카테고리 정보가 없으면 스킵
        if query_category not in db_category_counts:
            skipped_count += 1
            continue
        
        total_relevant = db_category_counts[query_category]
        
        # 거리 기준 Top-K 인덱스 추출 (이미 계산된 거리 사용)
        top_k_indices = np.argsort(distances[i])[:k]
        
        # 검색된 카테고리들
        retrieved_categories = db_categories[top_k_indices]
        
        # 관련성 판단
        is_relevant = (retrieved_categories == query_category)
        
        # 메트릭 계산
        metrics = calculate_metrics(is_relevant, total_relevant, k)
        
        results['mAP'].append(metrics['mAP'])
        results['NDCG'].append(metrics['NDCG'])
        results['Precision'].append(metrics['precision'])
        results['Recall'].append(metrics['recall'])
    
    if skipped_count > 0:
        print(f"  ⚠️ 경고: {skipped_count}개 쿼리가 카테고리 정보 부족으로 스킵됨")
    
    return results

# --- (6) 메인 함수 ---
def main():
    
    print("\n" + "="*80)
    print(f" [{MODEL_NAME}] Deep Learning Model Evaluation (Optimized)")
    print(f" Metrics: mAP@{TOP_K}, NDCG@{TOP_K}, Precision@{TOP_K}, Recall@{TOP_K}")
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

    # 3. DB 카테고리별 문서 수 계산
    print("\n[Step 2] 카테고리별 문서 수 계산 중...")
    db_category_counts = build_category_counts(db_filenames)
    print(f"  - 총 {len(db_category_counts)}개 카테고리 발견")
    
    # 4. 전체 쿼리 평가 (Euclidean)
    print(f"\n[Step 3] Euclidean Distance로 평가 중...")
    metrics_euclidean = evaluate_batch(
        query_features, query_filenames, 
        db_features, db_filenames, 
        db_category_counts, 
        metric='euclidean', 
        k=TOP_K
    )
    
    # 5. 전체 쿼리 평가 (Manhattan)
    print(f"\n[Step 4] Manhattan Distance로 평가 중...")
    metrics_manhattan = evaluate_batch(
        query_features, query_filenames, 
        db_features, db_filenames, 
        db_category_counts, 
        metric='manhattan', 
        k=TOP_K
    )

    # 6. 최종 평균 계산 및 출력
    if not metrics_euclidean['mAP']:
        print("❌ 오류: 유효한 평가 결과가 없습니다.")
        print("  - DB와 쿼리의 카테고리가 일치하는지 확인하세요.")
        return

    def print_metric_row(metric_name: str, eu_list: List[float], mh_list: List[float]):
        eu_mean = np.mean(eu_list)
        mh_mean = np.mean(mh_list)
        print(f"  {metric_name:<15} | {eu_mean:.6f}     | {mh_mean:.6f}")

    print("\n" + "="*70)
    print(f" 모델 [{MODEL_NAME}] 최종 성능표 (@{TOP_K})")
    print("-" * 70)
    print(f"  {'Metric':<15} | {'Euclidean':<12} | {'Manhattan':<12}")
    print("-" * 70)
    
    print_metric_row("mAP", metrics_euclidean['mAP'], metrics_manhattan['mAP'])
    print_metric_row("NDCG", metrics_euclidean['NDCG'], metrics_manhattan['NDCG'])
    print_metric_row("Precision", metrics_euclidean['Precision'], metrics_manhattan['Precision'])
    print_metric_row("Recall", metrics_euclidean['Recall'], metrics_manhattan['Recall'])
    
    print("="*70)
    
    # 백분율 표시
    print("\n" + "="*70)
    print(f"  {'Metric':<15} | {'Euclidean (%)':<14} | {'Manhattan (%)':<14}")
    print("-" * 70)
    
    for metric_name in ['mAP', 'NDCG', 'Precision', 'Recall']:
        eu_pct = np.mean(metrics_euclidean[metric_name]) * 100
        mh_pct = np.mean(metrics_manhattan[metric_name]) * 100
        print(f"  {metric_name:<15} | {eu_pct:>10.2f}%     | {mh_pct:>10.2f}%")
    
    print("="*70)
    print(" * 이 데이터를 논문 [Deep Learning Model] 성능 비교표에 사용하세요.")
    print(f" * 총 평가 쿼리 수: {len(metrics_euclidean['mAP'])}개")

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
            print("\n========> 사용법: python deep_evaluate_map_optimized.py [model_name]")
            print("                (예: python deep_evaluate_map_optimized.py efficientnet)")
            sys.exit(1)

        process_time(main)

    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()