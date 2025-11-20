import cv2
import numpy as np
import sqlite3
import glob
import json
from tqdm import tqdm
from contextlib import contextmanager
import multiprocessing as mp
from functools import partial

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import get_category_from_path_dir,extract_hu_moments,measure_process_time


SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
DB_PATH = config.DB_PATH_HUMOMENT
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

# 상수
HU_MOMENTS_DIM = 7
BATCH_SIZE = 100

@contextmanager
def get_db_connection(db_path):
    
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"데이터베이스 오류: {e}")
        raise
    finally:
        conn.close()

def setup_database(db_path):
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            image_path TEXT PRIMARY KEY,
            category TEXT,
            feature_vector TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed_at 
        ON features(processed_at)
        """)
    print(f"데이터베이스 설정 완료: {db_path}")

 
def get_processed_images(cursor):
    """... (기존과 동일) ..."""
    cursor.execute("SELECT image_path FROM features")
    return {row[0] for row in cursor.fetchall()}


def collect_image_paths():
    """... (기존과 동일) ..."""
    image_paths = []
    for directory in SOURCE_DIRS:
        for ext in IMAGE_EXTENSIONS:
            paths = glob.glob(os.path.join(directory, ext))
            image_paths.extend(paths)
            print(f"  └ {directory}에서 {len(paths)}개의 {ext} 파일 발견")
    return image_paths

def process_single_image(image_path, source_dirs):
    """단일 이미지 처리 (워커 프로세스용)"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        features = extract_hu_moments(image)
        category = get_category_from_path_dir(image_path, source_dirs)
        
        return (image_path, category, json.dumps(features.tolist()))
    except Exception as e:
        print(f"에러 ({image_path}): {e}")
        return None
def process_images_parallel(skip_existing=True):
    """
    멀티프로세싱을 사용한 병렬 처리
    """
    setup_database(DB_PATH)
    
    print("\n이미지 파일 목록 수집 중...")
    image_paths = collect_image_paths()
    print(f"총 {len(image_paths)}개의 이미지 발견\n")
    
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # 이미 처리된 이미지 필터링
        processed_images = get_processed_images(cursor) if skip_existing else set()
        if processed_images:
            print(f"{len(processed_images)}개의 이미지가 이미 처리됨\n")
        
        # 미처리 이미지만 선택
        images_to_process = [p for p in image_paths if p not in processed_images]
        print(f"처리할 이미지: {len(images_to_process)}개\n")
        
        # CPU 코어 수 결정 (전체 코어의 70% 사용)
        num_workers = max(1, int(mp.cpu_count() * 0.7))
        print(f"{num_workers}개의 워커 프로세스 사용\n")
        
        # 멀티프로세싱 풀 생성
        process_func = partial(process_single_image, source_dirs=SOURCE_DIRS)
        
        batch_data = []
        success_count = 0
        error_count = 0
        
        with mp.Pool(processes=num_workers) as pool:
            # 병렬 처리 + 진행률 표시
            for result in tqdm(
                pool.imap(process_func, images_to_process, chunksize=50),
                total=len(images_to_process),
                desc="Hu Moments 추출 중 (병렬)"
            ):
                if result is None:
                    error_count += 1
                    continue
                
                batch_data.append(result)
                
                # 배치 단위로 DB 저장
                if len(batch_data) >= BATCH_SIZE:
                    cursor.executemany("""
                    INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
                    VALUES (?, ?, ?)
                    """, batch_data)
                    conn.commit()
                    success_count += len(batch_data)
                    batch_data = []
        
        # 남은 데이터 저장
        if batch_data:
            cursor.executemany("""
            INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
            VALUES (?, ?, ?)
            """, batch_data)
            conn.commit()
            success_count += len(batch_data)
    
    # 최종 통계
    print("\n" + "=" * 50)
    print("처리 완료!")
    print(f"  성공: {success_count}개")
    print(f"  실패: {error_count}개")
    print("=" * 50)

def get_statistics():
    """... (기존과 동일) ..."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM features")
        total = cursor.fetchone()[0]
        print(f"\n데이터베이스에 저장된 특징 벡터 수: {total}")

if __name__ == "__main__":
    try:
         
        measure_process_time(lambda: process_images_parallel(skip_existing=True))
        get_statistics()
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {e}")
        raise