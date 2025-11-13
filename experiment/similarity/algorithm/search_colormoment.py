 

import sqlite3
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import json 

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import extract_color_moment_rgb,measure_process_time   


# ============================================================================
# 경로 및 K값 설정
# ============================================================================
DB_PATH = config.DB_PATH_COLORMOMENT
TOP_K = config.TOP_K

RELEVANT_CATEGORY = None
TARGET_IMAGE_PATH=None

# Recall@K 계산을 위한 데이터셋의 총 정답 개수
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP

# ============================================================================
# 품질 지표 계산 함수 
# ============================================================================
def calculate_metrics(
    results: List[Dict],
    relevant_category: str,
    total_relevant_count: int,
    k: int
) -> Dict[str, float]:
    """
    검색 결과(Top-K)를 기반으로 품질 지표를 계산.
    """
    
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
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'map_at_k': map_at_k,
        'ndcg_at_k': ndcg_at_k
    }


# ============================================================================
# [수정] Color Moment 특징 추출 (공통 모듈 사용)
# ============================================================================
def get_target_feature(image_path: str) -> np.ndarray:
    """
    타겟 이미지 파일을 읽어 공통 Color Moment 특징을 추출.
    DB 생성시 사용된 'extract_color_moment_rgb'와 동일한 함수를 사용.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read target image: {image_path}")
    
    # 공통 모듈의 함수를 사용하여 특징 추출
    return extract_color_moment_rgb(image)


# ============================================================================
# DB 로드
# ============================================================================
def load_features_from_db(db_path: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    DB에서 모든 특징 벡터를 로드
    """
    # ... (기존 코드와 동일) ...
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
 
    cursor.execute(f"SELECT image_path, category, feature_vector FROM features")
    rows = cursor.fetchall()
    conn.close()
 
    results = []
    for image_path, category, feature_str in rows:
        try:
            feature_vector = np.array(json.loads(feature_str))
        except:
            try:
                feature_str_clean = feature_str.strip('[]')
                feature_vector = np.array([float(x.strip()) for x in feature_str_clean.split(',')])
            except Exception as e:
                print(f"Warning: Failed to parse feature for {image_path}: {e}")
                print(f"Feature string sample: {feature_str[:100]}...")
                continue
 
        results.append((image_path, category, feature_vector))
 
    return results


# ============================================================================
# 거리 계산
# ============================================================================
def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """유클리디안 거리 계산"""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """맨하탄 거리 계산"""
    return np.sum(np.abs(vec1 - vec2))


# ============================================================================
# 유사도 검색
# ============================================================================
def find_similar_images(
    target_feature: np.ndarray,
    db_features: List[Tuple[str, str, np.ndarray]],
    metric: str = 'euclidean',
    top_k: int = TOP_K
) -> List[Dict]:
    """
    유사한 이미지 검색
    """
    # ... (기존 코드와 동일) ...
    distances = []
 
    distance_func = euclidean_distance if metric == 'euclidean' else manhattan_distance
 
    for image_path, category, feature_vector in db_features:
        if image_path == TARGET_IMAGE_PATH:
            continue
   
        distance = distance_func(target_feature, feature_vector)
        distances.append({
            'image_path': image_path,
            'category': category,
            'distance': distance
        })
 
    distances.sort(key=lambda x: x['distance'])
 
    top_results = distances[:top_k]
    for idx, item in enumerate(top_results, 1):
        item['rank'] = idx
 
    return top_results


# ============================================================================
# 결과 출력
# ============================================================================
def print_results(
    target_path: str,
    euclidean_results: List[Dict],
    manhattan_results: List[Dict],
    euclidean_metrics: Dict,
    manhattan_metrics: Dict,
    relevant_category: str
):
    """검색 결과와 품질 지표를 보기 좋게 출력"""
    # ... (기존 코드와 동일) ...
    print("\n" + "="*100)
    print("COLOR MOMENT SIMILARITY SEARCH RESULTS")
    print("="*100)
    print(f"Target Image: {target_path}")
    print(f"Relevant Category (Ground Truth): '{relevant_category}'")
    print("="*100)
 
    # --- Euclidean 결과 ---
    print(f"\nTop {TOP_K} Similar Images - EUCLIDEAN DISTANCE")
    print("-" * 100)
    for item in euclidean_results:
        print(f"Rank {item['rank']}:")
        print(f"  📁 Path:     {item['image_path']}")
        print(f"  🏷️  Category: {item['category']} {'<- [Relevant]' if item['category'] == relevant_category else ''}") # 정답 표시
        print(f"  📏 Distance: {item['distance']:.6f}")
        print()
 
    # --- Manhattan 결과 ---
    print(f"\nTop {TOP_K} Similar Images - MANHATTAN DISTANCE")
    print("-" * 100)
    for item in manhattan_results:
        print(f"Rank {item['rank']}:")
        print(f"  📁 Path:     {item['image_path']}")
        print(f"  🏷️  Category: {item['category']} {'<- [Relevant]' if item['category'] == relevant_category else ''}") # 정답 표시
        print(f"  📏 Distance: {item['distance']:.6f}")
        print()
 
    # --- 품질 지표(Metrics) 출력 ---
    print("\n" + "="*100)
    print(f"PERFORMANCE EVALUATION (K={TOP_K}, Relevant='{relevant_category}')")
    print("="*100)
 
    print(f"| {'Metric':<16} | {'Euclidean':<15} | {'Manhattan':<15} | {'Description'} |")
    print(f"|{'-'*18}|{'-'*17}|{'-'*17}|{'-'*33}|")
 
    def print_metric_row(metric_name, e_val, m_val, desc):
        print(f"| {metric_name:<16} | {e_val:<15.4f} | {m_val:<15.4f} | {desc} |")
 
    print_metric_row("Precision@K", 
                     euclidean_metrics['precision_at_k'], 
                     manhattan_metrics['precision_at_k'], 
                     "Top-K 중 정답 비율")
    print_metric_row("Recall@K", 
                     euclidean_metrics['recall_at_k'], 
                     manhattan_metrics['recall_at_k'], 
                     "전체 정답 중 찾은 비율 (N={})".format(TOTAL_RELEVANT_COUNT_MAP.get(relevant_category, 0))) # [수정] .get() 사용
    print_metric_row("mAP@K", 
                     euclidean_metrics['map_at_k'], 
                     manhattan_metrics['map_at_k'], 
                     "정답 순서 고려 (정밀도 평균)")
    print_metric_row("NDCG@K", 
                     euclidean_metrics['ndcg_at_k'], 
                     manhattan_metrics['ndcg_at_k'], 
                     "정답 순서 가중치 (1위에 높은 점수)")
 
    print("="*100)


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    """유사도 검색 및 품질 지표 계산 메인 함수"""
    
    print("\nStarting Color Moment Similarity Search...")
    print(f"Target Image: {TARGET_IMAGE_PATH}")
    print(f"Database: {DB_PATH}")
    print(f"Relevant Category: '{RELEVANT_CATEGORY}' (K={TOP_K})")
    
    # 1. 타겟 이미지 특징 추출
    print("\n[Step 1] Extracting Color Moment features from target image...")
    target_feature = get_target_feature(TARGET_IMAGE_PATH) # <--- [수정] 새 래퍼 함수 사용
    print(f"Feature vector extracted: shape={target_feature.shape}")
    
    # 2. DB에서 특징 벡터 로드
    print("\n[Step 2] Loading features from database...")
    db_features = load_features_from_db(DB_PATH)
    print(f"Loaded {len(db_features)} images from database")
    
    # 3. Euclidean 거리로 검색
    print("\n[Step 3] Searching with Euclidean distance...")
    euclidean_results = find_similar_images(
        target_feature, 
        db_features, 
        metric='euclidean', 
        top_k=TOP_K
    )
    print(f"Found top {TOP_K} similar images (Euclidean)")
    
    # 4. Manhattan 거리로 검색
    print("\n[Step 4] Searching with Manhattan distance...")
    manhattan_results = find_similar_images(
        target_feature, 
        db_features, 
        metric='manhattan', 
        top_k=TOP_K
    )
    print(f"Found top {TOP_K} similar images (Manhattan)")
    
    # 5. 품질 지표 계산
    print("\n[Step 5] Calculating performance metrics...")
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(RELEVANT_CATEGORY, 0)
    if total_relevant == 0:
        print(f"Warning: TOTAL_RELEVANT_COUNT_MAP에 '{RELEVANT_CATEGORY}' 키가 없습니다. Recall@K가 0이 됩니다.")
        
    euclidean_metrics = calculate_metrics(euclidean_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    manhattan_metrics = calculate_metrics(manhattan_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    print(f"Metrics calculated")

    # 6. 결과 및 지표 출력
    print_results(
        TARGET_IMAGE_PATH, 
        euclidean_results, 
        manhattan_results,
        euclidean_metrics,
        manhattan_metrics,
        RELEVANT_CATEGORY
    )
    
    return {
        'euclidean_results': euclidean_results,
        'manhattan_results': manhattan_results,
        'euclidean_metrics': euclidean_metrics,
        'manhattan_metrics': manhattan_metrics
    }


if __name__ == "__main__":
    try:
        
        if len(sys.argv) == 3:
            RELEVANT_CATEGORY = sys.argv[1]
            target_image_name= sys.argv[2]
            TARGET_IMAGE_PATH=f"{config.TARGET_IMAGE_ALGORITHM_DIR}/{target_image_name}.png"
        else :
            print("\n========> 첫번째 파라메터는 카테고리, 두번째 카테고리는 확장자 제외한 이미지 이름을 등록하세요.")
        
        results_data=measure_process_time(main)
        
        print("\nSearch and Evaluation completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()