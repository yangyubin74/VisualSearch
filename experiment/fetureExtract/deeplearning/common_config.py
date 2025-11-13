
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import random

import time 
from datetime import datetime

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import measure_process_time  as process_time


print("공통 설정 로드됨.")

# --- 1. 기본 경로 설정 (사용자 업데이트 경로) ---
BASE_IMAGE_DIR = Path(config.MODEL_BASE_IMAGE_DIR)
MODEL_SAVE_DIR = Path(config.MODEL_SAVE_DIR)
FEATURE_SAVE_DIR = Path(config.FEATURE_SAVE_DIR)

# --- 2. 상수 정의 ---
CLASSES =config.CLASSES
SPLITS = config.SPLITS

# 이미지 크기
IMG_SIZE_EFFICIENTNET = config.IMG_SIZE_EFFICIENTNET
IMG_SIZE_AE = config.IMG_SIZE_AE
IMG_SIZE_SIAMESE =config.IMG_SIZE_SIAMESE

IMAGE_EXTENSIONS=config.IMAGE_EXTENSIONS

# --- 3. 경로 생성 함수 ---
def create_directories():
    """필요한 모든 폴더를 생성."""
    # 모델 저장 폴더 생성
    (MODEL_SAVE_DIR / "efficientnet").mkdir(parents=True, exist_ok=True)
    (MODEL_SAVE_DIR / "autoencoder").mkdir(parents=True, exist_ok=True)
    (MODEL_SAVE_DIR / "siamesenetwork").mkdir(parents=True, exist_ok=True)

    # 특징 저장 폴더 생성
    (FEATURE_SAVE_DIR / "efficientnet").mkdir(parents=True, exist_ok=True)
    (FEATURE_SAVE_DIR / "autoencoder").mkdir(parents=True, exist_ok=True)
    (FEATURE_SAVE_DIR / "siamesenetwork").mkdir(parents=True, exist_ok=True)
    print("모델 및 특징 저장 폴더 확인/생성 완료.")

# --- 4. 이미지 경로 로드 함수 ---
def load_image_paths():
    """모든 이미지 경로를 스캔하고 딕셔너리로 반환."""
    image_paths = {}
    all_train_paths = []
    
    print("이미지 파일 스캔 시작...")
    
    for c in SPLITS:
        image_paths[c] = {}
        for s in CLASSES:
            current_paths = []
            search_dir = BASE_IMAGE_DIR / c / s
            for ext in IMAGE_EXTENSIONS:
                pattern = str(search_dir / ext)
                current_paths.extend(glob.glob(pattern))
            
            image_paths[c][s] = current_paths
            
            if s == 'train':
                all_train_paths.extend(current_paths)
                
    print(f"총 'dress' 학습 이미지: {len(image_paths['train']['dress'])}개")
    print(f"총 'pants' 학습 이미지: {len(image_paths['train']['pants'])}개")
    print(f"총 'shirt' 학습 이미지: {len(image_paths['train']['shirt'])}개")
    print(f"--- 총 학습 이미지: {len(all_train_paths)}개 ---")
    
    return image_paths, all_train_paths

# --- 5. 공통 특징 추출 함수 ---
def extract_features_from_paths(image_path_list, model, preprocess_func, target_size, desc="특징 추출 중"):
    """
    주어진 이미지 경로 리스트에서 특징을 추출하는 공용 함수
    """
    features_list = []
    
    for img_path in tqdm(image_path_list, desc=desc):
        try:
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_func(img_batch)
            features = model.predict(img_preprocessed, verbose=0)
            features_list.append(features.flatten())
        except Exception as e:
            print(f"파일 처리 오류 {img_path}: {e}")
            
    return np.array(features_list)

def measure_process_time(func):

    process_time(func)
     