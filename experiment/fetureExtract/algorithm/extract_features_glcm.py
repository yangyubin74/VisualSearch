import cv2
import numpy as np
import sqlite3
import os
import glob
import json
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import extract_glcm, measure_process_time, get_category_from_path_dir

# 설정 상수
GLCM_LEVELS = config.GLCM_LEVELS
BATCH_SIZE = 100
MIN_IMAGE_SIZE = 32
NUM_WORKERS = max(1, cpu_count() - 1)  # CPU 코어 수 - 1 (최소 1개)

SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
DB_PATH = config.DB_PATH_GLCM
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

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

def process_single_image(image_path, source_dirs, glcm_levels):
    
    try:
        # 이미지 읽기
        image = cv2.imread(image_path)
        
        # 이미지 유효성 검증
        is_valid, error_msg = validate_image(image, image_path)
        if not is_valid:
            return (image_path, None, None, error_msg)
        
        # GLCM 특징 추출
        features = extract_glcm(image, glcm_levels)
        features_json = json.dumps(features.tolist())
        
        # 카테고리 추출
        category = get_category_from_path_dir(image_path, source_dirs)
        
        return (image_path, category, features_json, None)
        
    except Exception as e:
        return (image_path, None, None, str(e))

def batch_insert_to_db(cursor, results):
    """배치 결과를 DB에 삽입"""
    success_count = 0
    error_count = 0
    
    for image_path, category, features_json, error_msg in results:
        if error_msg is None:
            try:
                cursor.execute("""
                INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
                VALUES (?, ?, ?)
                """, (image_path, category, features_json))
                success_count += 1
            except Exception as e:
                print(f"\nDB 저장 에러: {image_path} - {e}")
                error_count += 1
        else:
            error_count += 1
            
    return success_count, error_count

def process_images():
    """모든 이미지를 병렬로 처리하며 특징을 추출하고 DB에 저장."""
    print(f"데이터베이스 설정 중: {DB_PATH}")
    
    conn, cursor = setup_database(DB_PATH)
    
    try:
        print("이미지 파일 목록 수집 중...")
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
            
        total_images = len(image_paths)
        print(f"총 {total_images}개의 이미지 처리 시작...")
        print(f"병렬 처리 워커 수: {NUM_WORKERS}개")
        
        success_count = 0
        error_count = 0
        
        # partial 함수로 고정 인자 전달
        process_func = partial(
            process_single_image,
            source_dirs=SOURCE_DIRS,
            glcm_levels=GLCM_LEVELS
        )
        
        # 병렬 처리를 위한 프로세스 풀 생성
        with Pool(processes=NUM_WORKERS) as pool:
            # chunksize 설정: 전체 이미지 수를 워커 수로 나눈 값
            chunksize = max(1, total_images // (NUM_WORKERS * 4))
            
            # imap으로 순차적 결과 처리 + tqdm 진행바
            results_iterator = pool.imap(
                process_func,
                image_paths,
                chunksize=chunksize
            )
            
            batch_results = []
            
            for result in tqdm(results_iterator, total=total_images, desc="GLCM 추출 중"):
                batch_results.append(result)
                
                # 배치 크기만큼 모이면 DB에 저장
                if len(batch_results) >= BATCH_SIZE:
                    s_count, e_count = batch_insert_to_db(cursor, batch_results)
                    success_count += s_count
                    error_count += e_count
                    conn.commit()
                    batch_results = []
            
            # 남은 결과 처리
            if batch_results:
                s_count, e_count = batch_insert_to_db(cursor, batch_results)
                success_count += s_count
                error_count += e_count
        
        print("\n데이터베이스에 변경사항 저장 중...")
        conn.commit()
        
        print(f"\n처리 완료!")
        print(f"  - 성공: {success_count}개")
        print(f"  - 실패: {error_count}개")
        print(f"  - 총계: {total_images}개")
        print(f"  - 성공률: {success_count/total_images*100:.1f}%")
        
    except Exception as e:
        print(f"\n치명적 에러 발생: {e}")
        raise
        
    finally:
        conn.close()

if __name__ == "__main__":
    measure_process_time(process_images)