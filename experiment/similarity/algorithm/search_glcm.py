
import cv2
import numpy as np
import sqlite3
import json

from scipy.spatial.distance import euclidean, cityblock
from typing import List, Dict, Tuple

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import extract_glcm,measure_process_time   

# ============================================================================
# 환경설정 값들
# ============================================================================
DB_PATH = config.DB_PATH_GLCM
TOP_K = config.TOP_K
GLCM_LEVELS = config.GLCM_LEVELS 

RELEVANT_CATEGORY = None
TARGET_IMAGE_PATH=None

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
    
    r = [1 if item['category'] == relevant_category else 0 for item in results]
    
    k_actual = len(r)
    if k_actual == 0:
        return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'map_at_k': 0.0, 'ndcg_at_k': 0.0}

    # 2. Precision@K
    precision_at_k = np.sum(r) / k_actual
    
    # 3. Recall@K
    recall_at_k = np.sum(r) / total_relevant_count if total_relevant_count > 0 else 0.0
    
    # 4. mAP@K
    precisions = []
    relevant_count = 0
    for i in range(k_actual):
        if r[i] == 1:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
            
    map_at_k = np.mean(precisions) if precisions else 0.0
    
    # 5. NDCG@K
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


def get_target_features(image_path):
    """타겟 이미지의 GLCM 특징을 추출."""
    if not os.path.exists(image_path):
        print(f"오류: 타겟 이미지 파일을 찾을 수 없습니다. {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"오류: 타겟 이미지를 읽을 수 없습니다. {image_path}")
        return None
    
    print(f"타겟 이미지 '{os.path.basename(image_path)}' 특징 추출 중...")
    
    # [수정됨] 공용 함수 호출
    return extract_glcm(image, GLCM_LEVELS) 

# ============================================================================
# DB 로드
# ============================================================================
def load_features_from_db(db_path):
    """... (기존과 동일) ..."""
    if not os.path.exists(db_path):
        print(f"오류: 데이터베이스 파일을 찾을 수 없습니다. {db_path}")
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    db_features = []
    try:
        query = "SELECT image_path, category, feature_vector FROM features"
        cursor.execute(query)
        rows = cursor.fetchall()

        print(f"데이터베이스에서 {len(rows)}개의 특징 벡터 로드 중...")
        
        for row in rows:
            image_path, category, feature_str = row
            try:
                feature_vector = np.array(json.loads(feature_str))
                db_features.append((image_path, category, feature_vector))
            except json.JSONDecodeError:
                print(f"경고: {image_path}의 특징 벡터(JSON) 파싱 오류.")
                
    except sqlite3.Error as e:
        print(f"데이터베이스 조회 오류: {e}")
    finally:
        conn.close()
        
    return db_features

# ============================================================================
# 유사도 검색
# ============================================================================
def find_similar_images(
    target_features: np.ndarray,
    db_features: List[Tuple[str, str, np.ndarray]],
    target_image_path: str,
    top_k: int = TOP_K
) -> Tuple[List[Dict], List[Dict]]:
    
    distances = []
    
    for image_path, category, db_vec in db_features:
        if os.path.normpath(image_path) == os.path.normpath(target_image_path):
            continue
        
        dist_manhattan = cityblock(target_features, db_vec)
        dist_euclidean = euclidean(target_features, db_vec)
        
        distances.append({
            'image_path': image_path,
            'category': category,
            'dist_manhattan': dist_manhattan,
            'dist_euclidean': dist_euclidean
        })
        
    sorted_manhattan = sorted(distances, key=lambda x: x['dist_manhattan'])
    sorted_euclidean = sorted(distances, key=lambda x: x['dist_euclidean'])
    
    top_k_manhattan = sorted_manhattan[:top_k]
    top_k_euclidean = sorted_euclidean[:top_k]
    
    for i, item in enumerate(top_k_manhattan, 1):
        item['rank'] = i
        item['distance'] = item['dist_manhattan']

    for i, item in enumerate(top_k_euclidean, 1):
        item['rank'] = i
        item['distance'] = item['dist_euclidean']

    return top_k_manhattan, top_k_euclidean

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
    """... (기존과 동일) ..."""
    print("\n" + "="*100)
    print("GLCM SIMILARITY SEARCH RESULTS")
    print("="*100)
    print(f"Target Image: {target_path}")
    print(f"Relevant Category (Ground Truth): '{relevant_category}'")
    print("="*100)
    
    # --- Euclidean 결과 ---
    print(f"\nTop {TOP_K} Similar Images - EUCLIDEAN DISTANCE (L2)")
    print("-" * 100)
    for item in euclidean_results:
        print(f"Rank {item['rank']}:")
        print(f"  Path:     {item['image_path']}")
        print(f"  Category: {item['category']} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  Distance: {item['distance']:.6f}")
        print()
        
    # --- Manhattan 결과 ---
    print(f"\n Top {TOP_K} Similar Images - MANHATTAN DISTANCE (L1)")
    print("-" * 100)
    for item in manhattan_results:
        print(f"Rank {item['rank']}:")
        print(f"  Path:     {item['image_path']}")
        print(f"  Category: {item['category']} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  Distance: {item['distance']:.6f}")
        print()
        
    # --- 품질 지표(Metrics) 출력 ---
    print("\n" + "="*100)
    print(f"PERFORMANCE EVALUATION (K={TOP_K}, Relevant='{relevant_category}')")
    print("="*100)
    
    print(f"| {'Metric':<16} | {'Euclidean (L2)':<15} | {'Manhattan (L1)':<15} | {'Description'} |")
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
                     "전체 정답 중 찾은 비율 (N={})".format(TOTAL_RELEVANT_COUNT_MAP[relevant_category]))
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
    print("\nStarting GLCM Similarity Search...")
    print(f"Target Image: {TARGET_IMAGE_PATH}")
    print(f"Database: {DB_PATH}")
    print(f"Relevant Category: '{RELEVANT_CATEGORY}' (K={TOP_K})")

    # 1. 타겟 이미지 특징 추출
    print("\n[Step 1] Extracting GLCM features from target image...")
    target_features = get_target_features(TARGET_IMAGE_PATH)
    if target_features is None:
        return
    print(f"Feature vector extracted: shape={target_features.shape}")

    # 2. 데이터베이스에서 모든 특징 로드
    print("\n[Step 2] Loading features from database...")
    db_features = load_features_from_db(DB_PATH)
    if not db_features:
        print("데이터베이스에 특징 벡터가 없습니다. 프로그램을 종료합니다.")
        return
    print(f"Loaded {len(db_features)} images from database")

    print(f"\n[Step 3] 총 {len(db_features)}개의 이미지와 유사도 비교 시작...")
    
    # 3. 유사도 계산
    top_k_manhattan, top_k_euclidean = find_similar_images(
        target_features, 
        db_features,
        TARGET_IMAGE_PATH,
        top_k=TOP_K
    )
    print(f"Found top {TOP_K} similar images (Manhattan & Euclidean)")

    # 4. 품질 지표 계산
    print("\n[Step 4] Calculating performance metrics...")
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(RELEVANT_CATEGORY, 0)
    if total_relevant == 0:
        print(f"Warning: TOTAL_RELEVANT_COUNT_MAP에 '{RELEVANT_CATEGORY}' 키가 없습니다. Recall@K가 0이 됩니다.")
        
    euclidean_metrics = calculate_metrics(top_k_euclidean, RELEVANT_CATEGORY, total_relevant, TOP_K)
    manhattan_metrics = calculate_metrics(top_k_manhattan, RELEVANT_CATEGORY, total_relevant, TOP_K)
    print(f"Metrics calculated")

    # 5. 결과 출력
    print_results(
        TARGET_IMAGE_PATH,
        top_k_euclidean,
        top_k_manhattan,
        euclidean_metrics,
        manhattan_metrics,
        RELEVANT_CATEGORY
    )

if __name__ == "__main__":
      if len(sys.argv) == 3:
            RELEVANT_CATEGORY = sys.argv[1]
            target_image_name= sys.argv[2]
            TARGET_IMAGE_PATH=f"{config.TARGET_IMAGE_ALGORITHM_DIR}/{target_image_name}.png"
      else :
            print("\n========> 첫번째 파라메터는 카테고리, 두번째 카테고리는 확장자 제외한 이미지 이름을 등록하세요.")
        
      measure_process_time(main)

      print("\n Search and Evaluation completed successfully!")