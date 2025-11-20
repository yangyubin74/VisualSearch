"""
RGB 색공간 기반 Color Moment 특징 추출 및 DB 저장 (병렬처리)
"""
import cv2
import numpy as np
import sqlite3
import glob
import json
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from common_utility import extract_color_moment_rgb, measure_process_time, get_category_from_path_dir

# 설정 환경
SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS
DB_PATH = config.DB_PATH_COLORMOMENT


def setup_database(db_path):
    """데이터베이스와 테이블을 설정."""
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


def process_single_image(args):
    
    image_path, source_dirs = args
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        features = extract_color_moment_rgb(image)
        features_json = json.dumps(features.tolist())
        category = get_category_from_path_dir(image_path, source_dirs)
        
        return (image_path, category, features_json)
    
    except Exception as e:
        print(f"\n에러: {image_path} 처리 중 오류 발생 - {e}")
        return None


def save_to_database(results, db_path):
     
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # None이 아닌 결과만 필터링
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        cursor.executemany("""
        INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
        VALUES (?, ?, ?)
        """, valid_results)
        conn.commit()
    
    conn.close()
    return len(valid_results)


def process_images_parallel(num_workers=None):
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # 1개 코어는 시스템용으로 남김
    
    print(f"데이터베이스 설정 중: {DB_PATH}")
    setup_database(DB_PATH)
    
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
    print(f"병렬 처리 워커 수: {num_workers}")
    
    # 병렬처리를 위한 인자 준비
    args_list = [(img_path, SOURCE_DIRS) for img_path in image_paths]
    
    # 배치 크기 설정
    batch_size = 1000
    total_success = 0
    total_failed = 0
    
    # 배치 단위로 처리
    for i in range(0, len(args_list), batch_size):
        batch_args = args_list[i:i + batch_size]
        
        # 병렬처리로 특징 추출
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, batch_args),
                total=len(batch_args),
                desc=f"배치 {i//batch_size + 1}/{(len(args_list)-1)//batch_size + 1} 처리 중"
            ))
        
        # DB에 저장
        success_count = save_to_database(results, DB_PATH)
        failed_count = len(results) - success_count
        
        total_success += success_count
        total_failed += failed_count
        
        if failed_count > 0:
            print(f"\n배치 처리 완료 - 성공: {success_count}, 실패: {failed_count}")
    
    print(f"\n=== RGB Color Moment 특징 추출 완료! ===")
    print(f"총 이미지: {len(image_paths)}개")
    print(f"성공: {total_success}개")
    print(f"실패: {total_failed}개")
    print(f"DB 저장 위치: {DB_PATH}")

    


if __name__ == "__main__":
    # Windows에서 multiprocessing 사용 시 필요
    mp.freeze_support()
    
    # 병렬처리 버전 실행
    print("=== 병렬 처리 모드 ===")
    measure_process_time(lambda: process_images_parallel())
    