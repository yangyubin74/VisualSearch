import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Autoencoder는 'preprocess_input'을 사용하지 않음
from sklearn.metrics.pairwise import pairwise_distances
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config 
from common_utility import measure_process_time ,print_results,find_similar_images,calculate_metrics,extract_query_features,load_search_database,print_compact_table

# --- 설정값 ---
# 상위 몇 개를 반환할지
TOP_K = config.TOP_K
# 검색할 카테고리 (명령줄 인수로 받음)
RELEVANT_CATEGORY = None
# 타겟 이미지 경로 (명령줄 인수로 받음)
TARGET_IMAGE_PATH = None

# Recall@K 계산 시 분모로 사용될 전체 정답 개수
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP

# --- 1. [수정] 특징 추출 모델(인코더) 로드 ---
def load_encoder_model():
    
    model_path = Path(config.MODEL_SAVE_DIR) / "autoencoder"/config.SEED_DIR / "encoder_model.h5" 
    
    if not model_path.exists():
        print(f" 오류: 학습된 인코더 모델이 존재하지 않습니다: {model_path}")
        print(f"  파일이 해당 경로에 있는지, 파일명('encoder_model.h5')이 정확한지 확인하세요.")
        sys.exit(1)
    
    print(f" 특징 추출 모델(인코더) 로드 중: {model_path.name}")
    
    # 인코더 모델 자체를 직접 로드.
    feature_model = load_model(model_path)
    
    # 모델의 출력 차원을 확인.
    output_dim = feature_model.output.shape[-1]
    print(f"  특징 벡터 차원: {output_dim}")

    # DB의 특징 벡터 차원(256)과 일치하는지 확인
    if output_dim != 256:
        print(f" 경고: 로드된 인코더의 출력 차원({output_dim})이")
        print(f"    DB에 저장된 차원(Y.shape[1] == 256)과 일치하지 않습니다!")
                
    return feature_model
    


def main():
    """유사도 검색 및 품질 지표 계산 메인 함수"""
    
    print("\n Starting Autoencoder Similarity Search...") 
    print(f"  Target Image: {TARGET_IMAGE_PATH}")
    print(f"  Database: {Path(config.FEATURE_SAVE_DIR) / 'autoencoder' / config.SEED_DIR}") 
    print(f"  Relevant Category: '{RELEVANT_CATEGORY}' (K={TOP_K})")
    
    # 1. 특징 추출 모델(인코더) 로드
    print("\n[Step 1] Loading feature extractor model (Encoder)...")
    feature_model = load_encoder_model() # [수정]
    print(f" Model loaded")
    
    # 2. 타겟 이미지 특징 추출
    print("\n[Step 2] Extracting features from target image...")
    autoencoder_preprocess = lambda x: x.astype('float32') / 255.0
    query_features = extract_query_features(
        feature_model, 
        TARGET_IMAGE_PATH,
        config.IMG_SIZE_AE,
        preprocess_func=autoencoder_preprocess
     )
    print(f"Feature vector extracted: shape={query_features.shape}")
    
    # 3. DB에서 특징 벡터 로드 (전체 'train' 세트)
    print("\n[Step 3] Loading features from database...")
    db_features, db_filenames = load_search_database(split='train',model_name='autoencoder')
    print(f"Loaded {len(db_filenames)} images from database")
    
    # 4. Euclidean 거리로 검색
    print("\n[Step 4] Searching with Euclidean distance...")
    euclidean_results = find_similar_images(
        query_features, 
        db_features, 
        db_filenames, 
        metric='euclidean', 
        top_k=TOP_K
    )
    print(f"Found top {TOP_K} similar images (Euclidean)")
    
    # 5. Manhattan 거리로 검색
    print("\n[Step 5] Searching with Manhattan distance...")
    manhattan_results = find_similar_images(
        query_features, 
        db_features, 
        db_filenames, 
        metric='manhattan', 
        top_k=TOP_K
    )
    print(f"Found top {TOP_K} similar images (Manhattan)")
    
    # 6. 품질 지표 계산
    print("\n[Step 6] Calculating performance metrics...")
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(RELEVANT_CATEGORY, 0)
    if total_relevant == 0:
        print(f"Warning: TOTAL_RELEVANT_COUNT_MAP에 '{RELEVANT_CATEGORY}' 키가 없습니다. Recall@K가 0이 됩니다.")
        
    euclidean_metrics = calculate_metrics(euclidean_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    manhattan_metrics = calculate_metrics(manhattan_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    print(f"Metrics calculated")

    # 7. 결과 및 지표 출력 (main 함수 내부에서 호출)
    print_results(
        str(TARGET_IMAGE_PATH), 
        euclidean_results, 
        manhattan_results,
        euclidean_metrics,
        manhattan_metrics,
        RELEVANT_CATEGORY
    )

    # 7-1. 간결한 테이블 형식 출력
   

    print_compact_table(euclidean_results, "EUCLIDEAN", RELEVANT_CATEGORY)
    print_compact_table(manhattan_results, "MANHATTAN", RELEVANT_CATEGORY)

    # 8. colormoment와 동일하게 dict 반환
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
            target_image_name = sys.argv[2]
            
            if not hasattr(config, 'TARGET_IMAGE_DEEPLEARNING_DIR'):
                print(" 경고: 'config.py'에 'TARGET_IMAGE_DEEPLEARNING_DIR'가 없습니다.")
                print("  './target_images' 디렉토리로 대체합니다.")
                config.TARGET_IMAGE_DEEPLEARNING_DIR = Path('./target_images')
                config.TARGET_IMAGE_DEEPLEARNING_DIR.mkdir(exist_ok=True)
            
            TARGET_IMAGE_PATH = Path(f"{config.TARGET_IMAGE_DEEPLEARNING_DIR}/{target_image_name}.png")

        else:
            print("\n========> 사용법: python 03_autoencoder_search_eval.py [카테고리] [이미지명]")
            print("  [카테고리]: shirt, pants, dress 중 하나 (타겟 이미지의 실제 정답)")
            print("  [이미지명]: 확장자(.png)를 제외한 타겟 이미지 파일명")
            print("\n  예시: python 03_autoencoder_search_eval.py shirt my_test_shirt_01")
            sys.exit(1)
        
        # 카테고리 유효성 검사
        if RELEVANT_CATEGORY not in TOTAL_RELEVANT_COUNT_MAP:
            print(f" 오류: '{RELEVANT_CATEGORY}'는 유효한 카테고리가 아닙니다.")
            print(f" 사용 가능: {list(TOTAL_RELEVANT_COUNT_MAP.keys())}")
            sys.exit(1)

        # 타겟 이미지 존재 여부 확인
        if not TARGET_IMAGE_PATH.exists():
            print(f" 오류: 타겟 이미지를 찾을 수 없습니다.")
            print(f" 경로: {TARGET_IMAGE_PATH}")
            sys.exit(1)

        # 시간 측정 및 메인 함수 실행
        results_data = measure_process_time(main)

        # main 함수가 출력을 모두 처리했으므로, 여기서는 완료 메시지만 출력
        print("\nSearch and Evaluation completed successfully!")

    except Exception as e:
        print(f"\n스크립트 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()