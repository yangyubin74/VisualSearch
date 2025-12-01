from re import S
import cv2
import numpy as np
from scipy.stats import skew
import time 
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

from typing import List, Dict, Tuple
from pathlib import Path
import sys
import os
import config

import psutil
from dataclasses import dataclass
from typing import Optional

#============================실행시간 측정 공통함수[S]===================================
def measure_process_time(func):
    
    # 시작 시간 출력
    start_datetime = datetime.now()
    formatted_start = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start measurement =============>: {formatted_start}")

    # 처리 시간 측정
    total_start = time.time()
    func()  # 전달된 함수 실행
    total_elapsed = time.time() - total_start
    print(f"전체 특징점 추출 총 소요 시간: {total_elapsed:.2f}초")

    # 종료 시간 출력
    end_datetime = datetime.now()
    formatted_end = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(f"End measurement =============>: {formatted_end}")
    print("\n\n")
    print("===========Hello, this is Yubin Yang.=============================")
    print(f"Measurement Time:{formatted_start} ~ {formatted_end}")
    print("=========== I am Korean.==========================================")

#============================실행시간 측정 공통함수[E]===================================



#=====================Colormoment 특징점 추출과 이미지 검색 시 공동으로 사용[S] ===========
def extract_color_moment_rgb(image):
    
    if image is None:
        raise ValueError("입력 이미지가 None입니다.")
    
    # 1. BGR -> RGB 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. OpenCV 최적화 함수로 평균(Mean)과 표준편차(Std)를 한 번에 계산 (C++ 속도)
    # shape: (3, 1) -> flatten -> (3,)
    means, stds = cv2.meanStdDev(rgb_image)
    means = means.flatten()
    stds = stds.flatten()
    
    features = []
    
    # 3. 채널별 Skewness 벡터 연산 최적화
    for i in range(3):
        m = means[i]
        s = stds[i]
        
        # 표준편차가 0이면(단색 이미지) 왜도도 0
        if s == 0:
            skewness = 0.0
        else:
            # NumPy 벡터 연산으로 왜도 직접 계산 (Scipy보다 빠름)
            # 수식: E[ ((x-mu)/sigma)^3 ]
            channel = rgb_image[:, :, i]
            # (x - mean) 계산 비용 최소화
            skewness = np.mean(((channel - m) / s) ** 3)
            
        features.extend([m, s, skewness])
    
    return np.array(features)

#=====================Colormoment 특징점 추출과 이미지 검색 시 공동으로 사용[E] ===========


#=====================GLCM 특징점 추출과 이미지 검색 시 공동으로 사용[S] ==================
def extract_glcm(image, glcm_levels):
    
    
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GLCM 레벨 축소를 위한 이미지 리스케일링
    if glcm_levels < 256:
        # 0 ~ (glcm_levels-1) 범위로 정규화
        gray_image = (gray_image / 256.0 * glcm_levels).astype(np.uint8)
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(
        gray_image, 
        distances=[1], 
        angles=angles, 
        levels=glcm_levels,
        symmetric=True, 
        normed=True
    )
    
    # ASM과 energy는 동일하므로 ASM 제거
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    
    for prop in properties:
        prop_values = graycoprops(glcm, prop)
        features.extend([
            np.mean(prop_values),
            np.std(prop_values)  # 표준편차 추가
        ])
        
    return np.array(features)
#=====================GLCM 특징점 추출과 이미지 검색 시 공동으로 사용[E] =================


#=====================Hu Moment 특징점 추출과 이미지 검색 시 공동으로 사용[S] ============
LOG_EPSILON = 1e-10
def extract_hu_moments(image):

    #image = cv2.resize(image, config.IMG_SIZE_ALGORITHM, interpolation=cv2.INTER_AREA)

    """Hu Moments + 추가 형태 특징"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 1. Hu Moments (7차원)
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + LOG_EPSILON)
    log_hu[~np.isfinite(log_hu)] = 0
    
    # 2. 추가 형태 특징
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        
        # 면적 비율
        area_ratio = cv2.contourArea(largest) / (image.shape[0] * image.shape[1])
        
        # 둘레
        perimeter = cv2.arcLength(largest, True)
        
        # 종횡비
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # 원형도 (Circularity)
        circularity = 4 * np.pi * cv2.contourArea(largest) / (perimeter ** 2) if perimeter > 0 else 0
        
        # 볼록성 (Convexity)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        convexity = cv2.contourArea(largest) / hull_area if hull_area > 0 else 0
        
        additional_features = np.array([
            area_ratio, 
            np.log10(perimeter + 1),
            aspect_ratio, 
            circularity,
            convexity
        ])
    else:
        additional_features = np.zeros(5)
    
    # 통합 (7 + 5 = 12차원)
    return np.concatenate([log_hu, additional_features])

#=====================Hu Moment 특징점 추출과 이미지 검색 시 공동으로 사용[E] ============



# --- 이미지 경로에서 카테고리 추출 헬퍼 (공통) ---
def get_category_from_path_dir(image_path, source_dirs):
    """이미지 경로에서 카테고리를 추출합니다."""
    # ... (기존 코드와 동일) ...
    normalized_path = os.path.normpath(os.path.abspath(image_path))
    sorted_dirs = sorted(source_dirs, key=lambda x: len(x), reverse=True)
 
    for dir_path in sorted_dirs:
        normalized_dir = os.path.normpath(os.path.abspath(dir_path))
        try:
            relative = os.path.relpath(normalized_path, normalized_dir)
            if not relative.startswith('..'):
                return os.path.basename(normalized_dir)
        except ValueError:
            continue
 
    return "unknown"



TOP_K = config.TOP_K
TOTAL_RELEVANT_COUNT_MAP = config.TOTAL_RELEVANT_COUNT_MAP

#=====================검색 결과 출력 공통함수[S] ===========================
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
    print("FEATURE SIMILARITY SEARCH RESULTS") 
    print("="*100)
    print(f"Target Image: {Path(target_path).name}")
    print(f"Relevant Category (Ground Truth): '{relevant_category}'")
    print("="*100)
 
    # --- Euclidean 결과 ---
    print(f"\nTop {TOP_K} Similar Images - EUCLIDEAN DISTANCE (Lower is better)")
    print("-" * 100)
    for item in euclidean_results:
        print(f"Rank {item['rank']:<2}:")
        print(f"  Path:     {item['image_path']}")
        print(f"  Category: {item['category']:<8} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  Distance: {item['distance']:.6f}")
        print()
 
    # --- Manhattan 결과 ---
    print(f"\n Top {TOP_K} Similar Images - MANHATTAN DISTANCE (Lower is better)")
    print("-" * 100)
    for item in manhattan_results:
        print(f"Rank {item['rank']:<2}:")
        print(f"  Path:     {item['image_path']}")
        print(f"  Category: {item['category']:<8} {'<- [Relevant]' if item['category'] == relevant_category else ''}")
        print(f"  Distance: {item['distance']:.6f}")
        print()
 
    # --- 품질 지표(Metrics) 출력 ---
    print("\n" + "="*100)
    print(f"PERFORMANCE EVALUATION (K={TOP_K}, Relevant='{relevant_category}')")
    print("="*100)
 
    print(f"| {'Metric':<16} | {'Euclidean':<15} | {'Manhattan':<15} | {'Description'} |")
    print(f"|{'-'*18}|{'-'*17}|{'-'*17}|{'-'*33}|")
 
    def print_metric_row(metric_name, e_val, m_val, desc):
        print(f"| {metric_name:<16} | {e_val:<15.4f} | {m_val:<15.4f} | {desc} |")
 
    total_relevant = TOTAL_RELEVANT_COUNT_MAP.get(relevant_category, 0)

    print_metric_row("mAP@K", 
                     euclidean_metrics['map_at_k'], 
                     manhattan_metrics['map_at_k'], 
                     "정답 순서 고려 (정밀도 평균)")
    print_metric_row("NDCG@K", 
                     euclidean_metrics['ndcg_at_k'], 
                     manhattan_metrics['ndcg_at_k'], 
                     "정답 순서 가중치 (1위에 높은 점수)")

    print_metric_row("Precision@K", 
                     euclidean_metrics['precision_at_k'], 
                     manhattan_metrics['precision_at_k'], 
                     "Top-K 중 정답 비율")
    print_metric_row("Recall@K", 
                     euclidean_metrics['recall_at_k'], 
                     manhattan_metrics['recall_at_k'], 
                     f"전체 정답 중 찾은 비율 (N={total_relevant})")
 
    print("="*100)

#=====================검색 결과 출력 공통함수[E] ===========================


#=====================유사 이미지 검색 공통함수[S] ===========================
# --- 4. 카테고리 추출 헬퍼 ---
def get_category_from_path(path_str: str) -> str:
    """파일 경로에서 카테고리를 추출합니다."""
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
        
       
        results.append({
            "rank": rank,
            "image_path": file_path,
            "category": get_category_from_path(file_path),
            "distance": float(distances[idx])
        })
        
    return results
#=====================유사 이미지 검색 공통함수[E] ===========================

# ===================== 성능 지표 계산 헬퍼 함수 =====================
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



# ===================== 쿼리 이미지 특징점 추출 헬퍼 함수 =====================
def extract_query_features(model, img_path, img_size: Tuple[int, int], preprocess_func=None):
    
    try:
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 전처리 함수가 전달되었으면 적용, 없으면 그대로 사용 (혹은 에러 처리)
        if preprocess_func:
            img_preprocessed = preprocess_func(img_batch)
        else:
            # 기본값 혹은 경고
            print("경고: 전처리 함수가 지정되지 않았습니다.")
            img_preprocessed = img_batch

        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"오류: 쿼리 이미지 '{img_path}' 처리 실패: {e}")
        sys.exit(1)


# --- 이미지 검색 시 Feature DB 데이터 로드
def load_search_database(split='train',model_name=""):
    
    if model_name=="" :
        print("오류: 모델 이름이 제공되지 않았습니다.")
        sys.exit(1)
    
    print("=" * 25,"눈으로 확인","=" * 25)
    print(f"{model_name}의 {config.SEED_DIR} 디렉토리 파일을 로드 합니다.")
    print("=" * 60)

    feature_dir =Path(config.FEATURE_SAVE_DIR) / model_name / config.SEED_DIR
    
    all_db_features = []
    all_db_filenames = []
    
    print(f"전체 '{split}' 데이터베이스 로드 중...")
    
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
            print(f"  로드: {class_name} (특징 {db_features.shape[0]}개, 파일 {len(db_filenames)}개)")

        except Exception as e:
            print(f"오류: {class_name} 데이터베이스 로드 실패: {e}")
            sys.exit(1)

    if not all_db_filenames:
        print(f"  오류: '{split}' 스플릿에 대한 데이터베이스 파일이 전혀 없습니다.")
        print(f"  경로: {feature_dir}")
        sys.exit(1)

    final_db_features = np.vstack(all_db_features)
    
    print(f"\n 총 특징 벡터: {final_db_features.shape}")
    print(f"   총 파일명: {len(all_db_filenames)}개")

    if final_db_features.shape[0] != len(all_db_filenames):
        print("오류: 최종 .npy 파일과 .json 파일의 항목 수가 일치하지 않습니다!")
        sys.exit(1)
        
    return final_db_features, all_db_filenames

# ===================== 쿼리 이미지 특징점 추출 헬퍼 함수 끝 =====================



# ============================== FLOPS 측정 관련 함수 [S] ==============================
@dataclass
class FLOPSResult:
    """FLOPS 측정 결과를 저장하는 데이터 클래스"""
    total_operations: int  # 총 연산 횟수 (FLOPs)
    elapsed_time: float  # 소요 시간 (초)
    flops: float  # FLOPS (초당 연산 횟수)
    gflops: float  # GFLOPS (10억 연산/초)
    num_images: int  # 처리한 이미지 수
    flops_per_image: float  # 이미지당 FLOPS
    
    def __str__(self):
        return (
            f"{'='*60}\n"
            f"FLOPS 측정 결과\n"
            f"{'='*60}\n"
            f"총 연산 횟수:        {self.total_operations:,} FLOPs\n"
            f"소요 시간:          {self.elapsed_time:.2f}초\n"
            f"FLOPS:              {self.flops:,.0f} ops/sec\n"
            f"GFLOPS:             {self.gflops:.4f} GFLOPS\n"
            f"처리 이미지 수:      {self.num_images:,}개\n"
            f"이미지당 FLOPS:      {self.flops_per_image:,.0f} ops/sec\n"
            f"{'='*60}"
        )

class FLOPSCounter:
    """연산 횟수를 카운트하는 클래스"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """카운터 초기화"""
        self.total_ops = 0
        self.num_images = 0
    
    def count_color_moment_ops(self, image_shape):
        """Color Moment 연산 횟수 계산"""
        h, w, c = image_shape
        pixels = h * w
        
        ops = 0
        # 각 채널별로
        for _ in range(c):
            # Mean: (pixels-1)번의 덧셈 + 1번의 나눗셈
            ops += pixels + 1
            
            # Std: (pixels)번의 제곱 + (pixels-1)번의 덧셈 + 1번의 나눗셈 + 1번의 제곱근
            ops += pixels * 2 + pixels + 2
            
            # Skewness: 복잡한 연산 (근사치)
            # (x-mean)^3 계산: pixels * 3번 연산
            # 합계 및 정규화: pixels + 5번 정도
            ops += pixels * 4 + 5
        
        # RGB 변환 연산 (근사치)
        ops += pixels * 3
        
        self.total_ops += ops
        self.num_images += 1
        return ops
    
    def count_glcm_ops(self, image_shape, glcm_levels=256, num_angles=4):
        """GLCM 연산 횟수 계산"""
        h, w = image_shape[:2]
        pixels = h * w
        
        ops = 0
        
        # 그레이스케일 변환 (RGB -> Gray)
        ops += pixels * 3
        
        # GLCM 레벨 축소
        if glcm_levels < 256:
            ops += pixels * 2  # 나눗셈 + 곱셈
        
        # GLCM 행렬 계산 (각 각도별)
        # 근사치: pixels * 비교연산 * 각도 수
        ops += pixels * 2 * num_angles
        
        # GLCM 정규화
        ops += glcm_levels * glcm_levels * num_angles
        
        # 5가지 속성 계산 (contrast, dissimilarity, homogeneity, energy, correlation)
        # 각 속성은 GLCM 행렬 전체를 순회
        num_properties = 5
        ops += glcm_levels * glcm_levels * num_angles * num_properties * 3
        
        # 평균 및 표준편차 계산
        ops += num_properties * num_angles * 2
        
        self.total_ops += ops
        self.num_images += 1
        return ops
    
    def count_hu_moments_ops(self, image_shape):
        """Hu Moments 연산 횟수 계산"""
        h, w = image_shape[:2]
        pixels = h * w
        
        ops = 0
        
        # 그레이스케일 변환
        ops += pixels * 3
        
        # 이진화 (Otsu threshold)
        # 히스토그램 계산 + threshold 찾기
        ops += pixels + 256 * 10
        
        # Moments 계산 (m00, m10, m01, m20, m11, m02, m30, m21, m12, m03)
        # 각 moment는 픽셀 순회 + 곱셈/덧셈
        num_moments = 10
        ops += pixels * num_moments * 3
        
        # Hu Moments 계산 (7개)
        # 복잡한 수식이지만 근사치로 계산
        ops += 7 * 20  # 각 Hu moment당 약 20번의 연산
        
        # Log 변환
        ops += 7 * 2
        
        # 추가 형태 특징 (5개)
        # Contour 찾기 (근사치)
        ops += pixels * 2
        
        # 면적, 둘레, 종횡비, 원형도, 볼록성 계산
        ops += 100  # 근사치
        
        self.total_ops += ops
        self.num_images += 1
        return ops
    
    def get_result(self, elapsed_time: float) -> FLOPSResult:
        """측정 결과 반환"""
        if elapsed_time <= 0:
            flops = 0
            gflops = 0
            flops_per_image = 0
        else:
            flops = self.total_ops / elapsed_time
            gflops = flops / 1e9
            flops_per_image = flops / self.num_images if self.num_images > 0 else 0
        
        return FLOPSResult(
            total_operations=self.total_ops,
            elapsed_time=elapsed_time,
            flops=flops,
            gflops=gflops,
            num_images=self.num_images,
            flops_per_image=flops_per_image
        )


def measure_process_time_with_flops(func, flops_counter: Optional[FLOPSCounter] = None):
    """
    실행 시간과 FLOPS를 함께 측정하는 함수
    
    Args:
        func: 실행할 함수
        flops_counter: FLOPSCounter 인스턴스 (선택)
    
    Returns:
        FLOPSResult 또는 None (flops_counter가 없는 경우)
    """
    # 시작 시간 출력
    start_datetime = datetime.now()
    formatted_start = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start measurement =============>: {formatted_start}")

    # 처리 시간 측정
    total_start = time.time()
    result = func()  # 함수 실행
    total_elapsed = time.time() - total_start
    
    print(f"전체 특징점 추출 총 소요 시간: {total_elapsed:.2f}초")

    # 종료 시간 출력
    end_datetime = datetime.now()
    formatted_end = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print(f"End measurement =============>: {formatted_end}")
    
    # FLOPS 결과 출력
    if flops_counter is not None:
        flops_result = flops_counter.get_result(total_elapsed)
        print(f"\n{flops_result}")
        
        print("\n\n")
        print("===========Hello, this is Yubin Yang.=============================")
        print(f"Measurement Time:{formatted_start} ~ {formatted_end}")
        print("=========== I am Korean.==========================================")
        
        return flops_result
    else:
        print("\n\n")
        print("===========Hello, this is Yubin Yang.=============================")
        print(f"Measurement Time:{formatted_start} ~ {formatted_end}")
        print("=========== I am Korean.==========================================")
        return None

# ============================== FLOPS 측정 관련 함수 [E] ==============================
