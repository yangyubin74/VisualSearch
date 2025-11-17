import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cdist
from tqdm import tqdm # 진행상황 표시용

# ==========================================
# 1. 특징 추출 알고리즘 정의 (Feature Extractors)
# ==========================================

def extract_color_moments(img):
    """
    Color Moments: HSV 색상 공간에서 채널별 Mean, Std, Skewness 추출 (총 9차원)
    """
    if img is None: return None
    # RGB -> HSV 변환 (검색 성능이 보통 더 좋음)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    features = []
    # H, S, V 각 채널에 대해 루프
    for i in range(3):
        channel = hsv[:,:,i]
        features.append(np.mean(channel)) # 1st Moment: Mean
        features.append(np.std(channel))  # 2nd Moment: Std
        features.append(skew(channel.flatten())) # 3rd Moment: Skewness
        
    return np.array(features)

def extract_glcm(img):
    """
    GLCM: 그레이스케일 변환 후 텍스처 특징 추출 (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
    """
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # GLCM 행렬 계산 (거리 1, 각도 0, 45, 90, 135도 평균)
    # uint8 이미지는 256 레벨
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = []
    
    for prop in props:
        # 4개 각도의 평균값을 특징으로 사용
        val = graycoprops(glcm, prop).mean()
        features.append(val)
        
    return np.array(features)

def extract_hu_moments(img):
    """
    Hu Moments: 형태 불변 모멘트 7개 추출 (Log 변환 필수)
    """
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 7개의 Hu Moments 계산
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Log 변환 (스케일 보정) - 매우 중요!
    # 값이 너무 작아서 그대로 쓰면 거리 계산이 안됨
    # 공식: -sign(h) * log10(abs(h))
    processed_moments = []
    for num in hu_moments:
        if num == 0:
            processed_moments.append(0)
        else:
            processed_moments.append(-1 * np.sign(num) * np.log10(np.abs(num)))
            
    return np.array(processed_moments)

# ==========================================
# 2. mAP 계산 함수 (공통)
# ==========================================

def calculate_map_at_k(query_feats, gallery_feats, query_labels, gallery_labels, k=10, metric='euclidean'):
    """
    mAP@K를 계산하는 함수
    """
    # 1. 거리 행렬 계산 (Queries vs Gallery)
    # cdist는 매우 빠름. metric='euclidean' 또는 'cosine'
    dists = cdist(query_feats, gallery_feats, metric=metric)
    
    average_precisions = []
    
    # 각 쿼리에 대해 반복
    for i in range(len(dists)):
        # 거리 기준 오름차순 정렬 (가까운 순)
        sorted_indices = np.argsort(dists[i])
        
        # Top-K개만 자르기
        top_k_indices = sorted_indices[:k]
        
        # 검색된 이미지들의 라벨
        retrieved_labels = gallery_labels[top_k_indices]
        target_label = query_labels[i]
        
        # 정답(Relevant) 여부 확인 (True/False)
        is_relevant = (retrieved_labels == target_label)
        
        if np.sum(is_relevant) == 0:
            average_precisions.append(0)
            continue
            
        # Average Precision 계산
        score = 0.0
        num_hits = 0.0
        
        for rank, relevant in enumerate(is_relevant):
            if relevant:
                num_hits += 1
                # Precision @ Rank
                score += num_hits / (rank + 1)
                
        # GT(총 정답 수)로 나누기 (여기서는 Top-K 내의 mAP이므로 min(총정답수, K)로 나누기도 함)
        # 일반적인 검색 mAP는 '총 정답 개수'로 나눕니다.
        total_positives = np.sum(gallery_labels == target_label)
        # 자기 자신을 제외해야 한다면 total_positives - 1 처리가 필요할 수 있음
        
        average_precisions.append(score / min(total_positives, k))

    return np.mean(average_precisions)

# ==========================================
# 3. 실행 메인 코드
# ==========================================

def main():
    # -----------------------------------------------------------
    # [설정] 사용자의 데이터 경로로 수정하세요
    # -----------------------------------------------------------
    # 예: image_paths = ['data/img1.jpg', 'data/img2.jpg', ...]
    #     labels = [0, 0, 1, 2, ...] (클래스 ID)
    
    # ** 데모를 위한 가짜 데이터 생성 (실제 사용 시 삭제하세요) **
    print("Generating dummy data for demonstration...")
    num_images = 100
    image_paths = [f"dummy_{i}.jpg" for i in range(num_images)]
    labels = np.random.randint(0, 5, num_images) # 5개 클래스
    
    # 실제 이미지가 없으므로 랜덤 노이즈 이미지 생성
    dummy_images = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(num_images)]
    # -----------------------------------------------------------

    # 평가할 방법 리스트
    methods = ['Color-Moment', 'GLCM', 'Hu-Moment'] 
    # 딥러닝 피처가 있다면 여기에 'EfficientNet', 'MobileNetV3' 추가하고
    # 아래 루프에서 .npy 파일을 로드하도록 수정하면 됨

    results = {}

    for method in methods:
        print(f"\nProcessing Method: {method}...")
        features_list = []
        
        # 1. 특징 추출
        for i, img in enumerate(tqdm(dummy_images)): # 실제 사용시: cv2.imread(path) 사용
            # img = cv2.imread(image_paths[i]) # <--- 실제 코드
            
            if method == 'Color-Moment':
                feat = extract_color_moments(img)
            elif method == 'GLCM':
                feat = extract_glcm(img)
            elif method == 'Hu-Moment':
                feat = extract_hu_moments(img)
            
            features_list.append(feat)
            
        features = np.array(features_list)
        
        # 2. [중요] 데이터 정규화 (Min-Max Scaling)
        # 알고리즘 방식은 값의 범위가 제각각이라 거리 계산 전 필수!
        scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(features)
        
        # 3. mAP 계산
        # (자가 검색 Self-Retrieval 테스트: Query set = Gallery set)
        # 실제로는 Query set과 Gallery set을 나누는 것이 좋습니다.
        map_score = calculate_map_at_k(
            query_feats=features_normalized,
            gallery_feats=features_normalized,
            query_labels=labels,
            gallery_labels=labels,
            k=10, # Top-10 mAP
            metric='euclidean' # 알고리즘은 유클리디안 거리가 일반적
        )
        
        results[method] = map_score
        print(f"--> {method} mAP@10: {map_score:.4f}")

    print("\n================ Final Results ================")
    print(f"{'Method':<15} | {'mAP@10':<10}")
    print("-" * 30)
    for method, score in results.items():
        print(f"{method:<15} | {score:.4f}")

if __name__ == "__main__":
    main()