
import cv2
import numpy as np
import sqlite3
import os
import glob
import json
from tqdm import tqdm
from pathlib import Path

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import extract_glcm,measure_process_time   

# 설정 상수
GLCM_LEVELS = config.GLCM_LEVELS
BATCH_SIZE = 100
MIN_IMAGE_SIZE = 32

SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
DB_PATH =config.DB_PATH_GLCM
IMAGE_EXTENSIONS=config.IMAGE_EXTENSIONS

def setup_database(db_path):

    """데이터베이스와 테이블을 설정."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS features (
        image_path TEXT PRIMARY KEY,
        category TEXT,
        feature_vector TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_category ON features(category)
    """)
    
    conn.commit()
    return conn, cursor
 
def get_category_from_path(image_path, source_dirs):
    """이미지 경로에서 카테고리를 추출."""
    img_path = Path(image_path).resolve()
    
    for dir_path in source_dirs:
        dir_path_obj = Path(dir_path).resolve()
        try:
            img_path.relative_to(dir_path_obj)
            return dir_path_obj.name
        except ValueError:
            continue
            
    return "unknown"

def validate_image(image, image_path):
    """이미지 유효성을 검증."""
    if image is None:
        return False, "파일을 읽을 수 없습니다"
    
    if image.size == 0:
        return False, "빈 이미지입니다"
    
    height, width = image.shape[:2]
    if height < MIN_IMAGE_SIZE or width < MIN_IMAGE_SIZE:
        return False, f"이미지가 너무 작습니다 ({width}x{height})"
    
    return True, None

def process_images():
    """모든 이미지를 순회하며 특징을 추출하고 DB에 저장."""
    print(f"데이터베이스 설정 중: {DB_PATH}")
    
    with sqlite3.connect(DB_PATH) as conn:
        cursor = setup_database(DB_PATH)[1]
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        print("이미지 파일 목록 수집 중...")
        # ... (기존과 동일) ...
        image_paths = []
        for directory in SOURCE_DIRS:
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"경고: {directory} 디렉토리가 존재하지 않습니다.")
                continue
                
            for ext in IMAGE_EXTENSIONS:
                image_paths.extend(glob.glob(os.path.join(directory, ext)))
                
        if not image_paths:
            print("처리할 이미지가 없습니다.")
            return
            
        print(f"총 {len(image_paths)}개의 이미지 처리 시작...")
        
        success_count = 0
        error_count = 0
        
        for i, image_path in enumerate(tqdm(image_paths, desc="GLCM 추출 중")):
            try:
                image = cv2.imread(image_path)
                
                is_valid, error_msg = validate_image(image, image_path)
                if not is_valid:
                    print(f"\n경고: {image_path} - {error_msg}")
                    error_count += 1
                    continue
                
                # [수정됨] 공용 함수 호출
                features = extract_glcm(image, GLCM_LEVELS) 
                
                features_json = json.dumps(features.tolist())
                
                category = get_category_from_path(image_path, SOURCE_DIRS)
                
                cursor.execute("""
                INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
                VALUES (?, ?, ?)
                """, (image_path, category, features_json))
                
                success_count += 1
                
                if (i + 1) % BATCH_SIZE == 0:
                    conn.commit()
                    
            except Exception as e:
                print(f"\n에러: {image_path} 처리 중 오류 발생 - {e}")
                error_count += 1
                
        print("\n데이터베이스에 변경사항 저장 중...")
        conn.commit()
        
        print(f"\n처리 완료!")
        print(f"  - 성공: {success_count}개")
        print(f"  - 실패: {error_count}개")
        print(f"  - 총계: {len(image_paths)}개")

if __name__ == "__main__":

    measure_process_time(process_images)