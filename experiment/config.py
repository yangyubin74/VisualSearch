from pathlib import Path

#=============================유사도 검색 관련 옵션[S]=============================
#Target Image 디렉토리 경로 : 유사도 검색할 이미지 대상 지정 시 사용
TARGET_IMAGE_ALGORITHM_DIR = "/workspace/targetimage/algorithm"
TARGET_IMAGE_DEEPLEARNING_DIR = "/workspace/targetimage/deeplearning"
TARGET_IMAGE_SAMPLE_DIR = "/workspace/targetimage/sample"

# Recall@K 계산을 위한 데이터셋의 총 정답 개수: 유사도 검색 후 품질 지표 계산
TOTAL_RELEVANT_COUNT_MAP = {
    'shirt': 6821,
    'pants': 7096,
    'dress': 9181
}
#이미지 검색 Top 10개 조회 개수
TOP_K=10
#=============================유사도 검색 관련 옵션[E]=============================


#========================알고리즘  특징점 추출한 DB 경로[S]=========================
OUTPUT_DIR =  Path("/workspace/Journal/experiment/dataset/algorithm_features")

DB_PATH_COLORMOMENT = OUTPUT_DIR / "ColorMoment.db"
DB_PATH_GLCM= OUTPUT_DIR / "Glcm.db"
DB_PATH_HUMOMENT= OUTPUT_DIR / "Humemont.db"
DB_PATH_HUMOMENT_80= OUTPUT_DIR / "Humoment_80.db"
#========================알고리즘  특징점 추출한 DB 경로[E]=========================

#알고리즘용 이미지 리사이즈 크기
IMG_SIZE_ALGORITHM = (512, 512)

#GLCM 축소 Level 상수 값
GLCM_LEVELS=256


#이미지 확장자 유형
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']

#알고리즘 소스 이미지 경로
SOURCE_ALGORITHM_DIRS = [
    "/images/experimentimage/algorithm/dress",
    "/images/experimentimage/algorithm/pants",
    "/images/experimentimage/algorithm/shirt"
]


#========================딥러닝관련 옵션 [S]=========================
 # 1st : 42 , 2nd : 43, 3rd : 44
SEED=44
SEED_DIR = f"randomseed_{SEED}"



MODEL_SAVE_DIR="/workspace/Journal/experiment/dataset/model_train"
MODEL_BASE_IMAGE_DIR ="/images/experimentimage/model"
FEATURE_SAVE_DIR ="/workspace/Journal/experiment/dataset/model_features"

CLASSES = ['dress', 'pants', 'shirt']
SPLITS = ["train", "validation", "test"]

IMG_SIZE_EFFICIENTNET = (224, 224)
IMG_SIZE_AE = (128, 128)
IMG_SIZE_SIAMESE = (224, 224)
IMG_SIZE_MOBILENET=(224,224)

#========================딥러닝관련 옵션 [E]=========================