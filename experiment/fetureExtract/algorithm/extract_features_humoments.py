import cv2
import numpy as np
import sqlite3
import glob
import json
from tqdm import tqdm
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import time

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from common_utility import (
    get_category_from_path_dir,
    extract_hu_moments,
    measure_process_time,
    FLOPSCounter
)

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
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
    cursor.execute("SELECT image_path FROM features")
    return {row[0] for row in cursor.fetchall()}

def collect_image_paths():
    image_paths = []
    for directory in SOURCE_DIRS:
        if not os.path.exists(directory):
            continue
        for ext in IMAGE_EXTENSIONS:
            paths = glob.glob(os.path.join(directory, ext))
            image_paths.extend(paths)
    return image_paths

def process_single_image(image_path, source_dirs, shared_counter=None):
    """
    단일 이미지 처리 (워커 프로세스용) + FLOPS 측정
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, 0
        
        # FLOPS 카운트 (각 프로세스에서 개별 계산)
        ops_count = 0
        if shared_counter is not None:
            counter = FLOPSCounter()
            # common_utility에 count_hu_moments_ops 메서드가 구현되어 있어야 함
            ops_count = counter.count_hu_moments_ops(image.shape)
        
        features = extract_hu_moments(image)
        category = get_category_from_path_dir(image_path, source_dirs)
        
        return (image_path, category, json.dumps(features.tolist())), ops_count
        
    except Exception as e:
        print(f"에러 ({image_path}): {e}")
        return None, 0

from pathlib import Path

def process_images_parallel(skip_existing=True, enable_flops=True):
    """
    멀티프로세싱을 사용한 병렬 처리 및 FLOPS 측정
    Returns: total_operations
    """
    setup_database(DB_PATH)
    
    print("\n이미지 파일 목록 수집 중...")
    image_paths = collect_image_paths()
    print(f"총 {len(image_paths)}개의 이미지 발견\n")
    
    total_operations = 0
    
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # 이미 처리된 이미지 필터링
        processed_images = get_processed_images(cursor) if skip_existing else set()
        if processed_images:
            print(f"{len(processed_images)}개의 이미지가 이미 처리됨\n")
        
        # 미처리 이미지만 선택
        images_to_process = [p for p in image_paths if p not in processed_images]
        
        if not images_to_process:
            print("새로 처리할 이미지가 없습니다.")
            return 0
            
        print(f"실제 처리할 이미지: {len(images_to_process)}개\n")
        
        if enable_flops:
            print(f"FLOPS 측정: 활성화")

        # CPU 코어 수 결정 (전체 코어의 70% 사용)
        num_workers = max(1, int(mp.cpu_count() * 0.7))
        print(f"{num_workers}개의 워커 프로세스 사용\n")
        
        # 공유 카운터 설정 (멀티프로세싱용 - 실제 카운팅은 반환값으로 집계하지만 인터페이스 통일)
        manager = Manager()
        shared_counter = manager.Namespace() if enable_flops else None

        # 멀티프로세싱 풀 생성 및 인자 설정
        process_func = partial(
            process_single_image, 
            source_dirs=SOURCE_DIRS,
            shared_counter=shared_counter
        )
        
        batch_data = []
        success_count = 0
        error_count = 0
        batch_ops = 0
        
        with mp.Pool(processes=num_workers) as pool:
            # 병렬 처리 + 진행률 표시
            # imap 결과는 ((data), ops_count) 튜플임
            for result_tuple in tqdm(
                pool.imap(process_func, images_to_process, chunksize=50),
                total=len(images_to_process),
                desc="Hu Moments 추출 중 (병렬)"
            ):
                data, ops = result_tuple
                
                if data is None:
                    error_count += 1
                    continue
                
                batch_data.append(data)
                batch_ops += ops
                
                # 배치 단위로 DB 저장
                if len(batch_data) >= BATCH_SIZE:
                    cursor.executemany("""
                    INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
                    VALUES (?, ?, ?)
                    """, batch_data)
                    conn.commit()
                    
                    success_count += len(batch_data)
                    if enable_flops:
                        total_operations += batch_ops
                    
                    batch_data = []
                    batch_ops = 0
        
        # 남은 데이터 저장
        if batch_data:
            cursor.executemany("""
            INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
            VALUES (?, ?, ?)
            """, batch_data)
            conn.commit()
            success_count += len(batch_data)
            if enable_flops:
                total_operations += batch_ops
    
    # 최종 통계
    print("\n" + "=" * 50)
    print("처리 완료!")
    print(f"  성공: {success_count}개")
    print(f"  실패: {error_count}개")
    print("=" * 50)
    
    return total_operations

def get_statistics():
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM features")
        total = cursor.fetchone()[0]
        print(f"\n데이터베이스에 저장된 특징 벡터 수: {total}")

if __name__ == "__main__":
    # Windows multiprocessing 지원
    mp.freeze_support()
    
    print("=== Hu Moments 병렬 처리 모드 (FLOPS 측정 포함) ===")
    
    try:
        # FLOPS 카운터 생성
        flops_counter = FLOPSCounter()
        
        # 시간 측정 시작
        start_time = time.time()
        
        # 실행 및 FLOPS 집계
        # skip_existing=False로 하면 이미 있는 것도 다시 계산하여 FLOPS를 측정합니다.
        # 연구 목적의 측정이면 False 권장, 단순 DB 업데이트면 True
        total_ops = process_images_parallel(skip_existing=True, enable_flops=True)
        
        elapsed_time = time.time() - start_time
        
        # 통계 출력
        get_statistics()
        
        # FLOPS 결과 생성 및 출력
        if total_ops > 0:
            flops_counter.total_ops = total_ops
            flops_counter.num_images = 1 # 단순 비율 계산용
            
            print(f"\n{'='*60}")
            print("FLOPS 측정 결과")
            print(f"{'='*60}")
            print(f"총 연산 횟수:        {total_ops:,} FLOPs")
            print(f"소요 시간:          {elapsed_time:.2f}초")
            print(f"FLOPS:              {total_ops/elapsed_time:,.0f} ops/sec")
            print(f"GFLOPS:             {(total_ops/elapsed_time)/1e9:.4f} GFLOPS")
            print(f"{'='*60}")

    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {e}")
        raise

    print("\n\n")
    print("===========Hello, this is Yubin Yang.=============================")
    print("=========== I am Korean.==========================================")