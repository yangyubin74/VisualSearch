"""
RGB 색공간 기반 Color Moment 특징 추출 및 DB 저장
HSV 변환의 문제를 피하기 위해 RGB를 직접 사용합니다.
"""
import cv2
import numpy as np
import sqlite3

import glob
import json
from tqdm import tqdm
from pathlib import Path

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import extract_color_moment_rgb,measure_process_time   


# 설정 환경
SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
IMAGE_EXTENSIONS =config.IMAGE_EXTENSIONS
DB_PATH =config.DB_PATH_COLORMOMENT


def setup_database(db_path):
    """데이터베이스와 테이블을 설정합니다."""
    # ... (기존 코드와 동일) ...
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
 
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            image_path TEXT PRIMARY KEY,
            category TEXT, 
            feature_vector TEXT
        )
        """)
 
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_category ON features(category)
        """)
 
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        print(f"데이터베이스 설정 중 오류 발생: {e}")
        raise


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


def process_images():
    """모든 이미지를 순회하며 특징을 추출하고 DB에 저장합니다."""
    print(f"데이터베이스 설정 중: {DB_PATH}")
    conn, cursor = setup_database(DB_PATH)
    
    # ... (이미지 경로 수집 부분은 동일) ...
    print("이미지 파일 목록 수집 중...")
    image_paths = []
    for directory in SOURCE_DIRS:
        if not os.path.exists(directory):
            print(f"경고: 디렉토리가 존재하지 않습니다 - {directory}")
            continue
 
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
 
    if not image_paths:
        print("경고: 처리할 이미지가 없습니다.")
        return
 
    print(f"총 {len(image_paths)}개의 이미지 처리 시작...")
    
    batch_data = []
    batch_size = 100
    failed_count = 0
    
    for image_path in tqdm(image_paths, desc="RGB Color Moment 추출 중"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"\n경고: {image_path} 파일을 읽을 수 없습니다.")
                failed_count += 1
                continue
            
            # [수정] 공통 모듈의 함수 사용
            features = extract_color_moment_rgb(image) 
            features_json = json.dumps(features.tolist())
            
            category = get_category_from_path(image_path, SOURCE_DIRS)
            
            batch_data.append((image_path, category, features_json))
            
            if len(batch_data) >= batch_size:
                cursor.executemany("""
                INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
                VALUES (?, ?, ?)
                """, batch_data)
                conn.commit()
                batch_data = []
        
        except Exception as e:
            print(f"\n에러: {image_path} 처리 중 오류 발생 - {e}")
            failed_count += 1
    
    if batch_data:
        cursor.executemany("""
        INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
        VALUES (?, ?, ?)
        """, batch_data)
        conn.commit()
    
    conn.close()
    
    print(f"\nRGB Color Moment 특징 추출 완료!")
    print(f"성공: {len(image_paths) - failed_count}개")
    print(f"실패: {failed_count}개")
    print(f"DB 저장 위치: {DB_PATH}")


if __name__ == "__main__":

    measure_process_time(process_images)
    