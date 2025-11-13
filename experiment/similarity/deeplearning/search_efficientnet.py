import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import pairwise_distances
import json
import sys

from pathlib import Path
from typing import List, Dict, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import measure_process_time  as process_time 

# --- 설정값 ---
# 상위 몇 개를 반환할지
TOP_K = config.TOP_K
# 검색할 카테고리 (명령줄 인수로 받음)
RELEVANT_CATEGORY = None
# 타겟 이미지 경로 (명령줄 인수로 받음)
TARGET_IMAGE_PATH = None

# Recall@K 계산 시 분모로 사용될 전체 정답 개수
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP

# --- 1. 특징 추출 모델 로드 ---
def load_feature_extractor():
    """파인튜닝된 EfficientNet 모델을 로드하고 특징 추출기(neck)를 반환합니다."""

    model_path =Path(config.MODEL_SAVE_DIR) / "efficientnet" / "efficientnet_best.h5"
    if not model_path.exists():
        print(f"❌ 오류: 파인튜닝된 모델이 존재하지 않습니다: {model_path}")
        print(f"  '01b_efficientnet_finetune_extract.py'를 먼저 실행하세요.")
        sys.exit(1)
    
    print(f"📂 특징 추출 모델 로드 중: {model_path.name}")
    full_model = load_model(model_path)
    
    # 마지막 Dense 레이어(분류기)를 제거한 모델을 반환
    feature_model = Model(
        inputs=full_model.input, 
        outputs=full_model.layers[-2].output
    )
    print(f"  ✓ 특징 벡터 차원: {feature_model.output.shape[-1]}")
    return feature_model

# --- 2. 검색 대상 데이터베이스 로드 ---
def load_search_database(split='train'):
    """전체 'train' 스플릿의 특징 벡터와 파일명 리스트를 로드합니다."""
    feature_dir =Path(config.FEATURE_SAVE_DIR) / "efficientnet"
    
    all_db_features = []
    all_db_filenames = []
    
    print(f"📂 전체 '{split}' 데이터베이스 로드 중...")
    
    if not hasattr(config, 'CLASSES'):
        print("❌ 오류: common_config.py에 'CLASSES' 리스트가 정의되지 않았습니다.")
        sys.exit(1)

    for class_name in config.CLASSES:
        npy_path = feature_dir / f"{split}_{class_name}_features.npy"
        json_path = feature_dir / f"{split}_{class_name}_features.json"

        if not npy_path.exists() or not json_path.exists():
            print(f"  ⚠️ 경고: '{split}_{class_name}' 데이터베이스 파일을 찾을 수 없어 건너뜁니다.")
            continue

        try:
            db_features = np.load(npy_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                db_filenames = json.load(f)
            
            all_db_features.append(db_features)
            all_db_filenames.extend(db_filenames)
            print(f"  ✓ 로드: {class_name} (특징 {db_features.shape[0]}개, 파일 {len(db_filenames)}개)")

        except Exception as e:
            print(f"❌ 오류: {class_name} 데이터베이스 로드 실패: {e}")
            sys.exit(1)

    if not all_db_filenames:
        print(f"❌ 오류: '{split}' 스플릿에 대한 데이터베이스 파일이 전혀 없습니다.")
        print(f"  경로: {feature_dir}")
        print(f"  먼저 '01b_efficientnet_finetune_extract.py' 스크립트를 실행하세요.")
        sys.exit(1)

    final_db_features = np.vstack(all_db_features)
    
    print(f"\n  ✓ 총 특징 벡터: {final_db_features.shape}")
    print(f"  ✓ 총 파일명: {len(all_db_filenames)}개")

    if final_db_features.shape[0] != len(all_db_filenames):
        print("❌ 오류: 최종 .npy 파일과 .json 파일의 항목 수가 일치하지 않습니다!")
        sys.exit(1)
        
    return final_db_features, all_db_filenames

# --- 3. 쿼리 이미지 특징 추출 ---
def extract_query_features(model, img_path):
    """단일 쿼리 이미지의 특징 벡터를 추출합니다."""
    try:
        img = load_img(img_path, target_size=config.IMG_SIZE_EFFICIENTNET)
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"❌ 오류: 쿼리 이미지 '{img_path}' 처리 실패: {e}")
        sys.exit(1)

# --- 4. 카테고리 추출 헬퍼 ---
def get_category_from_path(path_str: str) -> str:
    """파일 경로에서 카테고리를 추출합니다."""
    try:
        return Path(path_str).parent.name
    except Exception:
        return "unknown"

# --- 5. [신규] 유사도 검색 (colormoment 구조와 유사하게) ---
def find_similar_images(
    target_feature: np.ndarray,
    db_features: np.ndarray,
    db_filenames: List[str],
    metric: str = 'euclidean',
    top_k: int = TOP_K
) -> List[Dict]:
    """
    지정된 메트릭(거리)을 사용하여 유사한 이미지를 검색합니다.
    (EfficientNet의 대규모 DB에 맞게 sklearn을 사용)
    """
    
    # 1. 거리 계산 (벡터화된 방식)
    query_features_2d = target_feature.reshape(1, -1)
    distances = pairwise_distances(
        query_features_2d, db_features, metric=metric
    ).flatten()
    
    # 2. Top-K 정렬 (거리가 *낮은* 순)
    top_indices = np.argsort(distances)[:top_k]
    
    # 3. 결과 포맷팅
    results = []
    for rank, idx in enumerate(top_indices, 1):
        file_path = db_filenames[idx]
        
        # 타겟 이미지 자체는 결과에서 제외 (colormoment와 동일 로직)
        # 참고: 이 로직은 Top-K를 뽑은 *후에* 적용하면 K개가 안될 수 있으므로
        #      colormoment와 달리 여기서는 포함합니다. 
        #      (대부분의 경우 타겟 이미지는 DB에 없음)
        # if file_path == str(TARGET_IMAGE_PATH):
        #     continue 
        
        results.append({
            "rank": rank,
            "image_path": file_path,
            "category": get_category_from_path(file_path),
            "distance": float(distances[idx])
        })
        
    return results

# --- 6. 성능 지표 계산 ---
def calculate_metrics(
    results: List[Dict], 
    relevant_category: str, 
    total_relevant_count: int, 
    k: int
) -> Dict:
    """Top-K 결과 리스트를 기반으로 성능 지표를 계산합니다."""
    
    # colormoment 스크립트의 calculate_metrics 함수와 동일한 로직
    r = [1 if item['category'] == relevant_category else 0 for item in results]
    k_actual = len(r)
    
    if k_actual == 0:
       return {'precision_at_k': 0.0, 'recall_at_k': 0.0, 'map_at_k': 0.0, 'ndcg_at_k': 0.0}

    # Precision@K
    precision_at_k = np.sum(r) / k_actual
    
    # Recall@K
    recall_at_k = np.sum(r) / total_relevant_count if total_relevant_count > 0 else 0.0
    
    # mAP@K
    precisions = []
    relevant_count = 0
    for i in range(k_actual):
        if r[i] == 1:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    map_at_k = np.mean(precisions) if precisions else 0.0
    
    # NDCG@K
    dcg = np.sum([r[i] / np.log2(i + 2) for i in range(k_actual)])
    ideal_r = sorted(r, reverse=True) # [1, 1, ..., 0, 0]
    idcg = np.sum([ideal_r[i] / np.log2(i + 2) for i in range(k_actual)])
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "map_at_k": map_at_k,
        "ndcg_at_k": ndcg_at_k
    }

# --- 7. 결과 출력 ---
def print_results(
    target_path: str,
    euclidean_results: List[Dict],
    manhattan_results: List[Dict],
    euclidean_metrics: Dict,
    manhattan_metrics: Dict,
    relevant_category: str
):
    """검색 결과와 품질 지표를 보기 좋게 출력"""
    print("\n" + "="*100)
    # 제목만 EfficientNet으로 변경, 그 외 형식은 colormoment와 동일
    print("EFFICIENTNET FEATURE SIMILARITY SEARCH RESULTS") 
    print("="*100)
    print(f"🎯 Target Image: {Path(target_path).name}")
    print(f"✅ Relevant Category (Ground Truth): '{relevant_category}'")
    print("="*100)
 
    # --- Euclidean 결과 ---
    print(f"\n📊 Top {TOP_K} Similar Images - EUCLIDEAN DISTANCE (Lower is better)")
    print("-" * 100)
    for item in euclidean_results:
        print(f"Rank {item['rank']:<2}:")
        print(f"  📁 Path:     {item['image_path']}")
        print(f"  🏷️  Category: {item['category']:<8} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  📏 Distance: {item['distance']:.6f}")
        print()
 
    # --- Manhattan 결과 ---
    print(f"\n📊 Top {TOP_K} Similar Images - MANHATTAN DISTANCE (Lower is better)")
    print("-" * 100)
    for item in manhattan_results:
        print(f"Rank {item['rank']:<2}:")
        print(f"  📁 Path:     {item['image_path']}")
        print(f"  🏷️  Category: {item['category']:<8} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  📏 Distance: {item['distance']:.6f}")
        print()
 
    # --- 품질 지표(Metrics) 출력 ---
    print("\n" + "="*100)
    print(f"📈 PERFORMANCE EVALUATION (K={TOP_K}, Relevant='{relevant_category}')")
    print("="*100)
 
    print(f"| {'Metric':<16} | {'Euclidean':<15} | {'Manhattan':<15} | {'Description'} |")
    print(f"|{'-'*18}|{'-'*17}|{'-'*17}|{'-'*33}|")
 
    def print_metric_row(metric_name, e_val, m_val, desc):
        print(f"| {metric_name:<16} | {e_val:<15.4f} | {m_val:<15.4f} | {desc} |")
 
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(relevant_category, 0)

    print_metric_row("Precision@K", 
                     euclidean_metrics['precision_at_k'], 
                     manhattan_metrics['precision_at_k'], 
                     "Top-K 중 정답 비율")
    print_metric_row("Recall@K", 
                     euclidean_metrics['recall_at_k'], 
                     manhattan_metrics['recall_at_k'], 
                     f"전체 정답 중 찾은 비율 (N={total_relevant})")
    print_metric_row("mAP@K", 
                     euclidean_metrics['map_at_k'], 
                     manhattan_metrics['map_at_k'], 
                     "정답 순서 고려 (정밀도 평균)")
    print_metric_row("NDCG@K", 
                     euclidean_metrics['ndcg_at_k'], 
                     manhattan_metrics['ndcg_at_k'], 
                     "정답 순서 가중치 (1위에 높은 점수)")
 
    print("="*100)

# --- 8. [수정] 메인 실행 (colormoment 구조) ---
def main():
    """유사도 검색 및 품질 지표 계산 메인 함수"""
    
    print("\n🚀 Starting EfficientNet Similarity Search...")
    print(f"📁 Target Image: {TARGET_IMAGE_PATH}")
    print(f"🎯 Relevant Category: '{RELEVANT_CATEGORY}' (K={TOP_K})")
    
    # 1. 특징 추출 모델 로드
    print("\n[Step 1] Loading feature extractor model...")
    feature_model = load_feature_extractor()
    print(f"✅ Model loaded")
    
    # 2. 타겟 이미지 특징 추출
    print("\n[Step 2] Extracting features from target image...")
    query_features = extract_query_features(feature_model, TARGET_IMAGE_PATH)
    print(f"✅ Feature vector extracted: shape={query_features.shape}")
    
    # 3. DB에서 특징 벡터 로드 (전체 'train' 세트)
    print("\n[Step 3] Loading features from database...")
    db_features, db_filenames = load_search_database(split='train')
    print(f"✅ Loaded {len(db_filenames)} images from database")
    
    # 4. Euclidean 거리로 검색
    print("\n[Step 4] Searching with Euclidean distance...")
    euclidean_results = find_similar_images(
        query_features, 
        db_features, 
        db_filenames, 
        metric='euclidean', 
        top_k=TOP_K
    )
    print(f"✅ Found top {TOP_K} similar images (Euclidean)")
    
    # 5. Manhattan 거리로 검색
    print("\n[Step 5] Searching with Manhattan distance...")
    manhattan_results = find_similar_images(
        query_features, 
        db_features, 
        db_filenames, 
        metric='manhattan', 
        top_k=TOP_K
    )
    print(f"✅ Found top {TOP_K} similar images (Manhattan)")
    
    # 6. 품질 지표 계산
    print("\n[Step 6] Calculating performance metrics...")
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(RELEVANT_CATEGORY, 0)
    if total_relevant == 0:
        print(f"Warning: TOTAL_RELEVANT_COUNT_MAP에 '{RELEVANT_CATEGORY}' 키가 없습니다. Recall@K가 0이 됩니다.")
        
    euclidean_metrics = calculate_metrics(euclidean_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    manhattan_metrics = calculate_metrics(manhattan_results, RELEVANT_CATEGORY, total_relevant, TOP_K)
    print(f"✅ Metrics calculated")

    # 7. 결과 및 지표 출력 (main 함수 내부에서 호출)
    print_results(
        str(TARGET_IMAGE_PATH), 
        euclidean_results, 
        manhattan_results,
        euclidean_metrics,
        manhattan_metrics,
        RELEVANT_CATEGORY
    )
    
    # 8. colormoment와 동일하게 dict 반환
    return {
        'euclidean_results': euclidean_results,
        'manhattan_results': manhattan_results,
        'euclidean_metrics': euclidean_metrics,
        'manhattan_metrics': manhattan_metrics
    }

# --- [수정] 스크립트 실행 (colormoment 구조) ---
if __name__ == "__main__":
    try:
        if len(sys.argv) == 3:
            RELEVANT_CATEGORY = sys.argv[1]
            target_image_name = sys.argv[2]
            
            if not hasattr(config, 'TARGET_IMAGE_DEEPLEARNING_DIR'):
                print("⚠️ 경고: 'common_config.py'에 'TARGET_IMAGE_DEEPLEARNING_DIR'가 없습니다.")
                print("  './target_images' 디렉토리로 대체합니다.")
                config.TARGET_IMAGE_DEEPLEARNING_DIR = Path('./target_images')
                config.TARGET_IMAGE_DEEPLEARNING_DIR.mkdir(exist_ok=True)
            
            TARGET_IMAGE_PATH = Path(f"{config.TARGET_IMAGE_DEEPLEARNING_DIR}/{target_image_name}.png")

        else:
            print("\n========> 사용법: python 02_efficientnet_search_eval.py [카테고리] [이미지명]")
            print("  [카테고리]: shirt, pants, dress 중 하나 (타겟 이미지의 실제 정답)")
            print("  [이미지명]: 확장자(.png)를 제외한 타겟 이미지 파일명")
            print("\n  예시: python 02_efficientnet_search_eval.py shirt my_test_shirt_01")
            sys.exit(1)
        
        # 카테고리 유효성 검사
        if RELEVANT_CATEGORY not in TOTAL_RELEVANT_COUNT_MAP:
            print(f"❌ 오류: '{RELEVANT_CATEGORY}'는 유효한 카테고리가 아닙니다.")
            print(f"  사용 가능: {list(TOTAL_RELEVANT_COUNT_MAP.keys())}")
            sys.exit(1)

        # 타겟 이미지 존재 여부 확인
        if not TARGET_IMAGE_PATH.exists():
            print(f"❌ 오류: 타겟 이미지를 찾을 수 없습니다.")
            print(f"  경로: {TARGET_IMAGE_PATH}")
            sys.exit(1)

        # 시간 측정 및 메인 함수 실행
        results_data = process_time(main)
        # if hasattr(config, 'process_time'):
        #     results_data = process_time(main)
        # else:
        #     print("경고: 'process_time' 함수를 찾을 수 없습니다. 시간 측정 없이 실행합니다.")
        #     results_data = main()

        # main 함수가 출력을 모두 처리했으므로, 여기서는 완료 메시지만 출력
        print("\n🎉 Search and Evaluation completed successfully!")

    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()