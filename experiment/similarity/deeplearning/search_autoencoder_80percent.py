import numpy as np
import cv2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Autoencoder는 'preprocess_input'을 사용하지 않음
from sklearn.metrics.pairwise import pairwise_distances
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    import config
    from common_utility import measure_process_time
except ImportError:
    print("오류: config.py 또는 common_utility.py를 찾을 수 없음.")
    sys.exit(1)

# ============================================================================
# --- 1. 환경설정 ---
# ============================================================================
TOP_K = 10 
TARGET_IMAGE_PATH = None

# ============================================================================
# --- 2. Autoencoder (인코더) 모델 로드 ---
# ============================================================================
def load_encoder_model() -> Model:
    """학습된 Autoencoder의 'Encoder' 모델을 로드."""
    
    # 'encoder_model.h5' 파일을 직접 지정
    model_path = Path(config.MODEL_SAVE_DIR) / "autoencoder" / "encoder_model.h5" 
    
    if not model_path.exists():
        print(f"오류: 학습된 인코더 모델이 존재하지 않습니다: {model_path}")
        sys.exit(1)
    
    print(f"특징 추출 모델(인코더) 로드 중: {model_path.name}")
    
    feature_model = load_model(model_path)
    print(f" 특징 벡터 차원: {feature_model.output.shape[-1]}")
    return feature_model

# ============================================================================
# --- 3. Autoencoder DB (80%) 로드 ---
# ============================================================================
def load_search_database_ae(split='train') -> Tuple[np.ndarray, List[str]]:
    
    feature_dir = Path(config.FEATURE_SAVE_DIR) / "autoencoder"
    
    all_db_features = []
    all_db_filenames = []
    
    print(f"📂 전체 '{split}' 데이터베이스(80%) 로드 중 (Autoencoder)...")
    
    if not hasattr(config, 'CLASSES'):
        print("오류: config.py에 'CLASSES' 리스트가 정의되지 않았습니다.")
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
        sys.exit(1)

    final_db_features = np.vstack(all_db_features)
    
    print(f"\n  총 특징 벡터: {final_db_features.shape}")
    print(f"    총 파일명: {len(all_db_filenames)}개 (80% DB)")

    if final_db_features.shape[0] != len(all_db_filenames):
        print(" 오류: 최종 .npy 파일과 .json 파일의 항목 수가 일치하지 않습니다!")
        sys.exit(1)
        
    return final_db_features, all_db_filenames

# ============================================================================
# --- 4. 쿼리 이미지 특징 추출 (AE용) ---
# ============================================================================
def extract_query_features_ae(model: Model, img_path: str) -> np.ndarray:
    try:
        # 128x128 크기 사용
        img = load_img(img_path, target_size=config.IMG_SIZE_AE) 
        img_array = img_to_array(img)
        
        img_normalized = img_array / 255.0 
        
        img_batch = np.expand_dims(img_normalized, axis=0)
        features = model.predict(img_batch, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"오류: 쿼리 이미지 '{img_path}' 처리 실패: {e}")
        sys.exit(1)

# ============================================================================
# --- 5. 공통 함수 (경로, 검색) ---
# ============================================================================
def get_category_from_path(path_str: str) -> str:
    
    try:
    
        return Path(path_str).parent.name
    except Exception:
        return "unknown"

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

# ============================================================================
# --- 6. 랭킹 출력 ---
# ============================================================================
def print_rankings(
    target_path: str,
    euclidean_results: List[Dict],
    manhattan_results: List[Dict]
):
    """검색된 Top-K 랭킹 리스트만 출력"""
    print("\n" + "="*100)
    print(f"AUTOENCODER 80% DB - SIMILARITY RANKING (TOP {TOP_K})")
    print("="*100)
    print(f"Target Image: {target_path}")
    print(f"Database: Autoencoder 'train' (80%) DB")
    print("="*100)
    
    print(f"\nTop {TOP_K} Similar Images - EUCLIDEAN DISTANCE (L2)")
    print("-" * 100)
    for item in euclidean_results:
        print(f"Rank {item['rank']:<3}: (Cat: {item['category']:<8}) {item['image_path']} (Dist: {item['distance']:.6f})")
        
    print(f"\nTop {TOP_K} Similar Images - MANHATTAN DISTANCE (L1)")
    print("-" * 100)
    for item in manhattan_results:
        print(f"Rank {item['rank']:<3}: (Cat: {item['category']:<8}) {item['image_path']} (Dist: {item['distance']:.6f})")
        
    print("="*100)

# ============================================================================
# --- 7. 메인 함수 ---
# ============================================================================
def main():
    """단순 랭킹 검색 메인 함수"""
    
    print("\n" + "="*60)
    print(f"  Autoencoder 80% DB 랭킹 검색 (Top-{TOP_K})")
    print(f"  Target: {TARGET_IMAGE_PATH}")
    print("="*60)
    
    # 1. 특징 추출 모델 로드
    print("\n[Step 1] Loading encoder model...")
    feature_model = load_encoder_model()
    print("Model loaded")
    
    # 2. 타겟 이미지 특징 추출
    print("\n[Step 2] Extracting features from target image...")
    query_features = extract_query_features_ae(feature_model, TARGET_IMAGE_PATH)
    print(f"Feature vector extracted: shape={query_features.shape}")
    
    # 3. DB에서 특징 벡터 로드 (전체 'train' 세트)
    print("\n[Step 3] Loading features from database(80%)...")
    db_features, db_filenames = load_search_database_ae('train')
    print(f"Loaded {len(db_filenames)} images from database")
    
    # 4. Euclidean 거리로 검색
    print("\n[Step 4] Searching with Euclidean distance...")
    euclidean_results = find_similar_images(
        query_features, db_features, db_filenames, 
        'euclidean', TOP_K
    )
    print(f"Found top {TOP_K} similar images (Euclidean)")
    
    # 5. Manhattan 거리로 검색
    print("\n[Step 5] Searching with Manhattan distance...")
    manhattan_results = find_similar_images(
        query_features, db_features, db_filenames, 
        'manhattan', TOP_K
    )
    print("Found top {TOP_K} similar images (Manhattan)")
    
    # 6. 랭킹 리스트 출력
    print_rankings(
        TARGET_IMAGE_PATH, 
        euclidean_results, 
        manhattan_results
    )
    
    return euclidean_results, manhattan_results

# ============================================================================
# --- 8. 스크립트 실행 ---
# ============================================================================
if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        target_image_name_no_ext = sys.argv[1]
        target_image_name = f"{target_image_name_no_ext}.png"
        
        # AE는 딥러닝 모델이므로, 딥러닝/샘플 경로에서 찾음
        path_dl = Path(config.TARGET_IMAGE_DEEPLEARNING_DIR) / target_image_name
        if not path_dl.exists():
            path_dl = Path(config.TARGET_IMAGE_DEEPLEARNING_DIR) / target_image_name

        if not path_dl.exists():
            print(f"❌ 오류: 다음 경로들에서 타겟 이미지를 찾을 수 없습니다:")
            print(f"  1. {config.TARGET_IMAGE_DEEPLEARNING_DIR / target_image_name}")

            sys.exit(1)
        
        TARGET_IMAGE_PATH = str(path_dl)
            
    else :
        print("  [이미지명]: 확장자(.png)를 제외한 타겟 이미지 파일명")
        sys.exit(1)

    measure_process_time(main)