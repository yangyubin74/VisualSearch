# search_humoment.py (수정됨)

import cv2
import numpy as np
import sqlite3
import json
from scipy.spatial import distance
from tqdm import tqdm
import time
from typing import List, Dict, Tuple

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import extract_hu_moments,measure_process_time

# ============================================================================
# 환경설정 값들
# ============================================================================
DB_PATH = config.DB_PATH_HUMOMENT
TOP_K = config.TOP_K

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
    """... (기존과 동일) ..."""
    # 1. Relevance List (r) 생성 (정답=1, 오답=0)
    r = [1 if item['category'] == relevant_category else 0 for item in results]
    
    k_actual = len(r)
    if k_actual == 0:
        return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'map_at_k': 0.0, 'ndcg_at_k': 0.0}

    # ... (이하 동일) ...
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

# --- 검색 로직 ---

def load_all_features_from_db(db_path):
    """... (기존과 동일) ..."""
    if not os.path.exists(db_path):
        print(f"오류: 데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"데이터베이스에서 특징 벡터 로드 중: {db_path}")
    
    query = "SELECT image_path, category, feature_vector FROM features"
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    db_features = []
    
    for row in tqdm(rows, desc="DB 특징 벡터 파싱 중"):
        image_path, category, features_json = row
        try:
            feature_vector = np.array(json.loads(features_json))
            db_features.append((image_path, category, feature_vector))
        except json.JSONDecodeError:
            print(f"경고: {image_path}의 특징 벡터(JSON)를 파싱하는 데 실패했습니다.")
            
    print(f"총 {len(db_features)}개의 특징 벡터 로드 완료.")
    return db_features

def search_similar_images(
    target_features: np.ndarray,
    db_features: List[Tuple[str, str, np.ndarray]],
    top_k: int,
    distance_metric: str = 'euclidean'
) -> List[Dict]:
     # [수정] 특징 벡터 차원에 따라 가중치 자동 설정
    feature_dim = len(target_features)
    
    if feature_dim == 7:
        # Hu Moments만 (7차원)
        weights = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    elif feature_dim == 10:
        # Hu Moments + 3개 추가 특징
        weights = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    elif feature_dim == 12:
        # Hu Moments + 5개 추가 특징
        weights = np.array([3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    else:
        # 알 수 없는 차원 - 균등 가중치
        weights = np.ones(feature_dim)
        print(f"⚠️  경고: 예상치 못한 특징 차원 ({feature_dim}). 균등 가중치 사용.")
    
    print(f"\n--- {distance_metric.capitalize()} 거리 계산 시작 (가중치 적용, 차원: {feature_dim}) ---")
    distances = []
    
    for image_path, category, feature_vector in db_features:
        if os.path.normpath(image_path) == os.path.normpath(TARGET_IMAGE_PATH):
            continue
        
        # 가중치 적용
        weighted_target = target_features * weights
        weighted_db = feature_vector * weights
        
        if distance_metric == 'euclidean':
            dist = distance.euclidean(weighted_target, weighted_db)
        elif distance_metric == 'manhattan':
            dist = distance.cityblock(weighted_target, weighted_db)
        elif distance_metric == 'cosine':
            dist = distance.cosine(weighted_target, weighted_db)
        else:
            raise ValueError("지원되지 않는 거리 지표입니다.")
            
        distances.append({
            'image_path': image_path,
            'category': category,
            'distance': dist
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
    """... (기존과 동일) ..."""
    print("\n" + "="*100)
    print("HU MOMENT SIMILARITY SEARCH RESULTS")
    print("="*100)
    print(f"🎯 Target Image: {target_path}")
    print(f"✅ Relevant Category (Ground Truth): '{relevant_category}'")
    print("="*100)
    
    # ... (이하 동일) ...
    # --- Euclidean 결과 ---
    print(f"\n📊 Top {TOP_K} Similar Images - EUCLIDEAN DISTANCE (L2)")
    print("-" * 100)
    for item in euclidean_results:
        print(f"Rank {item['rank']}:")
        print(f"  📁 Path:     {item['image_path']}")
        print(f"  🏷️  Category: {item['category']} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  📏 Distance: {item['distance']:.6f}")
        print()
        
    # --- Manhattan 결과 ---
    print(f"\n📊 Top {TOP_K} Similar Images - MANHATTAN DISTANCE (L1)")
    print("-" * 100)
    for item in manhattan_results:
        print(f"Rank {item['rank']}:")
        print(f"  📁 Path:     {item['image_path']}")
        print(f"  🏷️  Category: {item['category']} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  📏 Distance: {item['distance']:.6f}")
        print()
        
    # --- 품질 지표(Metrics) 출력 ---
    print("\n" + "="*100)
    print(f"📈 PERFORMANCE EVALUATION (K={TOP_K}, Relevant='{relevant_category}')")
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

def main():
    
      
    print("--- Hu Moment 유사도 검색 시작 ---")
    print(f"타겟 이미지: {TARGET_IMAGE_PATH}")
    print(f"데이터베이스: {DB_PATH}")
    print(f"🎯 Relevant Category: '{RELEVANT_CATEGORY}' (K={TOP_K})")
    print("-" * 30)

    # 1. 타겟 이미지 유효성 검사 및 로드
    if not os.path.exists(TARGET_IMAGE_PATH):
        print(f"오류: 타겟 이미지 파일을 찾을 수 없습니다: {TARGET_IMAGE_PATH}")
        exit()
        
    target_image = cv2.imread(TARGET_IMAGE_PATH)
    if target_image is None:
        print(f"오류: 타겟 이미지 파일을 읽을 수 없습니다: {TARGET_IMAGE_PATH}")
        exit()

    # 2. 타겟 이미지 특징 추출 (공용 함수 사용)
    print("\n[Step 1] 타겟 이미지 특징 추출 중...")
    target_features = extract_hu_moments(target_image) # <--- [수정됨] 공용 함수 호출
    print("✅ 타겟 이미지 특징 추출 완료.")

    # 3. 데이터베이스에서 모든 특징 로드
    print("\n[Step 2] 데이터베이스에서 모든 특징 로드 중...")
    db_features_list = load_all_features_from_db(DB_PATH)
    
    if not db_features_list:
        print("데이터베이스에서 특징을 로드하지 못했거나 DB가 비어있습니다. 프로그램을 종료합니다.")
        exit()

    # 4. 유사도 검색 수행 (Euclidean)
    print("\n[Step 3] 유사도 검색 수행 (Euclidean & Manhattan)...")
    euclidean_results = search_similar_images(
        target_features, 
        db_features_list, 
        top_k=TOP_K, 
        distance_metric='euclidean'
    )
    
    # 5. 유사도 검색 수행 (Manhattan)
    manhattan_results = search_similar_images(
        target_features, 
        db_features_list, 
        top_k=TOP_K, 
        distance_metric='manhattan'
    )
    print("✅ 유사도 검색 완료.")

    # 6. 품질 지표 계산
    print("\n[Step 4] Calculating performance metrics...")
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(RELEVANT_CATEGORY, 0)
    if total_relevant == 0:
        print(f"Warning: TOTAL_RELEVANT_COUNT_MAP에 '{RELEVANT_CATEGORY}' 키가 없습니다. Recall@K가 0이 됩니다.")
        
    euclidean_metrics = calculate_metrics(euclidean_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    manhattan_metrics = calculate_metrics(manhattan_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    print(f"✅ Metrics calculated")

    # 7. 결과 출력
    print_results(
        TARGET_IMAGE_PATH,
        euclidean_results,
        manhattan_results,
        euclidean_metrics,
        manhattan_metrics,
        RELEVANT_CATEGORY
    )

        
    print("\n🎉 Search and Evaluation completed successfully!")
# --- 메인 실행 ---

if __name__ == "__main__":
   
    if len(sys.argv) == 3:
        RELEVANT_CATEGORY = sys.argv[1]
        target_image_name= sys.argv[2]
        TARGET_IMAGE_PATH=f"{config.TARGET_IMAGE_ALGORITHM_DIR}/{target_image_name}.png"
    else :
        print("\n========> 첫번째 파라메터는 카테고리, 두번째 카테고리는 확장자 제외한 이미지 이름을 등록하세요.")


    measure_process_time(main)