# search_siamese.py 수정본

import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# [수정] EfficientNet 전처리 함수 임포트
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from sklearn.metrics.pairwise import pairwise_distances
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config 
from common_utility import measure_process_time, print_results, find_similar_images, calculate_metrics, extract_query_features, load_search_database

# --- 설정값 ---
TOP_K = config.TOP_K
RELEVANT_CATEGORY = None
TARGET_IMAGE_PATH = None
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP

# [삭제] 단순 나누기 함수 제거
# def preprocess_siamese_query(img_array):
#     return img_array.astype('float32') / 255.0

def load_feature_extractor():
    """학습된 Siamese Network의 'Base Network' 모델을 로드."""
    # ... (기존 코드와 동일) ...
    model_path = Path(config.MODEL_SAVE_DIR) / "siamesenetwork" / config.SEED_DIR / "base_network_best.h5" 
    
    if not model_path.exists():
        print(f"  파일이 해당 경로에 있는지, 파일명('base_network_best.h5')이 정확한지 확인하세요.")
        sys.exit(1)
    
    print(f"특징 추출 모델(Base Network) 로드 중: {model_path.name}")
    feature_model = load_model(model_path)
    
    # ... (출력 차원 확인 등 기존 로직 유지) ...
    if len(feature_model.output.shape) > 2:
        feature_model = Model(inputs=feature_model.input, outputs=feature_model.layers[-2].output)
        
    return feature_model

def main():
    """유사도 검색 및 품질 지표 계산 메인 함수"""
    
    print("\n Starting Siamese Network Similarity Search...")
    print(f" Target Image: {TARGET_IMAGE_PATH}")
    print(f" Database: {Path(config.FEATURE_SAVE_DIR) / 'siamesenetwork' / config.SEED_DIR}")
    print(f" Relevant Category: '{RELEVANT_CATEGORY}' (K={TOP_K})")
    
    # 1. 모델 로드
    print("\n[Step 1] Loading feature extractor model (Base Network)...") 
    feature_model = load_feature_extractor() 
    print(f"Model loaded")
    
    # 2. 타겟 이미지 특징 추출
    print("\n[Step 2] Extracting features from target image...")
    
    
    query_features = extract_query_features(
        feature_model, 
        TARGET_IMAGE_PATH,
        config.IMG_SIZE_SIAMESE,
        preprocess_func=preprocess_input   
    )
    print(f"Feature vector extracted: shape={query_features.shape}")
    
    # 3. DB 로드
    print("\n[Step 3] Loading features from database...")
    # model_name도 정확히 맞춰줍니다 (config.py 폴더명 기준 'siamesenetwork')
    db_features, db_filenames = load_search_database(split='train', model_name="siamesenetwork")
    print(f"Loaded {len(db_filenames)} images from database")
    
    # ... (이하 4, 5, 6, 7 단계는 기존과 동일) ...
    
    # 4. Euclidean 거리로 검색
    print("\n[Step 4] Searching with Euclidean distance...")
    euclidean_results = find_similar_images(
        query_features, db_features, db_filenames, metric='euclidean', top_k=TOP_K
    )
    
    # 5. Manhattan 거리로 검색
    print("\n[Step 5] Searching with Manhattan distance...")
    manhattan_results = find_similar_images(
        query_features, db_features, db_filenames, metric='manhattan', top_k=TOP_K
    )
    
    # 6. 지표 계산
    print("\n[Step 6] Calculating performance metrics...")
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(RELEVANT_CATEGORY, 0)
    
    euclidean_metrics = calculate_metrics(euclidean_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    manhattan_metrics = calculate_metrics(manhattan_results, RELEVANT_CATEGORY, total_relevant, TOP_K)

    # 7. 출력
    print_results(
        str(TARGET_IMAGE_PATH), 
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
    # ... (기존 메인 실행 로직 동일) ...
    try:
        if len(sys.argv) == 3:
            RELEVANT_CATEGORY = sys.argv[1]
            target_image_name = sys.argv[2]
            # ...
            TARGET_IMAGE_PATH = Path(f"{config.TARGET_IMAGE_DEEPLEARNING_DIR}/{target_image_name}.png")
        else:
            print("Usage Error...")
            sys.exit(1)
        
        if RELEVANT_CATEGORY not in TOTAL_RELEVANT_COUNT_MAP:
            sys.exit(1)
        if not TARGET_IMAGE_PATH.exists():
            sys.exit(1)

        measure_process_time(main)
        print("\n Search and Evaluation completed successfully!")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()