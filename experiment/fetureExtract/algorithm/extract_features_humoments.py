# extract_features_humoments.py (수정됨)

import cv2
import numpy as np
import sqlite3
import glob
import json
from tqdm import tqdm
from contextlib import contextmanager

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import get_category_from_path,extract_hu_moments,measure_process_time


SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
DB_PATH = config.DB_PATH_HUMOMENT
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS

# 상수
HU_MOMENTS_DIM = 7
BATCH_SIZE = 100

@contextmanager
def get_db_connection(db_path):
    """... (기존과 동일) ..."""
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"❌ 데이터베이스 오류: {e}")
        raise
    finally:
        conn.close()

def setup_database(db_path):
    """... (기존과 동일) ..."""
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
    print(f"✓ 데이터베이스 설정 완료: {db_path}")

 
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


def process_images(skip_existing=True):
    """..."""
    setup_database(DB_PATH)
    
    print("\n이미지 파일 목록 수집 중...")
    image_paths = collect_image_paths()
    print(f"✓ 총 {len(image_paths)}개의 이미지 발견\n")
    
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        
        processed_images = get_processed_images(cursor) if skip_existing else set()
        if processed_images:
            print(f"✓ {len(processed_images)}개의 이미지가 이미 처리됨\n")
        
        batch_data = []
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for image_path in tqdm(image_paths, desc="Hu Moments 추출 중"):
            if image_path in processed_images:
                skipped_count += 1
                continue
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"\n⚠ 경고: 이미지를 읽을 수 없음: {image_path}")
                    error_count += 1
                    continue
                
                # [수정됨] 공용 함수 호출
                features = extract_hu_moments(image) 
                features_json = json.dumps(features.tolist())
                
                # [!] 중요: 버그 수정
                # category를 먼저 계산해야 합니다.
                category = get_category_from_path(image_path, SOURCE_DIRS)
                
                # batch_data에 (path, category, feature) 튜플을 저장해야 합니다.
                batch_data.append((image_path, category, features_json))
                
                # 배치가 가득 찼으면 DB에 저장
                if len(batch_data) >= BATCH_SIZE:
                    # [!] 중요: 버그 수정
                    # executemany는 쿼리와 튜플 리스트만 받습니다.
                    cursor.executemany("""
                    INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
                    VALUES (?, ?, ?)
                    """, batch_data) # <--- 수정됨
                    
                    conn.commit()
                    success_count += len(batch_data)
                    batch_data = []
                    
            except Exception as e:
                print(f"\n❌ 에러: 이미지 처리 실패 ({image_path}): {e}")
                error_count += 1
        
        # 남은 데이터 저장
        if batch_data:
            # [!] 중요: 버그 수정
            cursor.executemany("""
            INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
            VALUES (?, ?, ?)
            """, batch_data) # <--- 수정됨
            
            conn.commit()
            success_count += len(batch_data)
    
    # 최종 통계
    print("\n" + "=" * 50)
    print("처리 완료!")
    print(f"  성공: {success_count}개")
    print(f"  스킵: {skipped_count}개")
    print(f"  실패: {error_count}개")
    print("=" * 50)


def get_statistics():
    """... (기존과 동일) ..."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM features")
        total = cursor.fetchone()[0]
        print(f"\n✓ 데이터베이스에 저장된 특징 벡터 수: {total}")

if __name__ == "__main__":
    try:
        measure_process_time(process_images)
        get_statistics()
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류 발생: {e}")
        raise