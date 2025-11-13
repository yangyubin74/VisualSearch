import cv2
import numpy as np
from scipy.stats import skew
import time 
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
import os
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
        raise ValueError("입력 이미지가 None입니다. 파일 경로를 확인하세요.")
        
    # BGR -> RGB 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb_image)
    channels = [r, g, b]
    
    features = []
    for channel in channels:
        channel_flat = channel.ravel()
        
        # 픽셀이 없는 경우(e.g., 마스킹된 이미지) 방지
        if channel_flat.size == 0:
            features.extend([0.0, 0.0, 0.0])
            continue
            
        mean = np.mean(channel_flat)
        std = np.std(channel_flat)
        skewness = skew(channel_flat)
        
        # skewness가 nan일 경우 (e.g., 모든 픽셀 값이 동일) 0으로 처리
        if np.isnan(skewness):
            skewness = 0.0
            
        features.extend([mean, std, skewness])
    
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





def get_category_from_path(image_path, source_dirs):
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