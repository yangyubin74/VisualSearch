
import cv2
import numpy as np
import sqlite3
import glob
import json
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial # partial 임포트 추가

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from common_utility import (
    extract_color_moment_rgb, 
    get_category_from_path_dir,
    FLOPSCounter
)

# 설정 환경
SOURCE_DIRS = config.SOURCE_ALGORITHM_DIRS
IMAGE_EXTENSIONS = config.IMAGE_EXTENSIONS
DB_PATH = config.DB_PATH_COLORMOMENT

# 최적화: 배치 사이즈를 100으로 줄여 응답성 향상
BATCH_SIZE = 100 

def setup_database(db_path):
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON features(category)")
        conn.commit()
        return conn, cursor
    except sqlite3.Error as e:
        print(f"데이터베이스 설정 중 오류 발생: {e}")
        raise

def process_single_image(args):
    """
    단일 이미지 처리 + FLOPS 카운팅 (Manager 제거로 인자 단순화)
    Args:
        args: (image_path, source_dirs, enable_flops) 튜플
    """
    image_path, source_dirs, enable_flops = args
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, 0
        
        # FLOPS 카운트 (로컬 계산 후 반환)
        ops_count = 0
        if enable_flops:
            # 매번 객체 생성하는 오버헤드를 줄이기 위해 단순 계산 권장하지만,
            # 현재 구조 유지를 위해 클래스 사용 (단, 공유 객체 아님)
            counter = FLOPSCounter()
            ops_count = counter.count_color_moment_ops(image.shape)
        
        features = extract_color_moment_rgb(image)
        features_json = json.dumps(features.tolist())
        category = get_category_from_path_dir(image_path, source_dirs)
        
        return (image_path, category, features_json), ops_count
    
    except Exception as e:
        # 에러 발생 시 로그는 줄이거나 파일로 빼는 것이 속도에 좋음
        return None, 0

def save_to_database(results, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    valid_results = [r[0] for r in results if r[0] is not None]
    
    if valid_results:
        cursor.executemany("""
        INSERT OR REPLACE INTO features (image_path, category, feature_vector) 
        VALUES (?, ?, ?)
        """, valid_results)
        conn.commit()
    
    conn.close()
    return len(valid_results)

def process_images_parallel(num_workers=None, enable_flops=True):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"데이터베이스 설정 중: {DB_PATH}")
    setup_database(DB_PATH)
    
    print("이미지 파일 목록 수집 중...")
    image_paths = []
    for directory in SOURCE_DIRS:
        if not os.path.exists(directory): continue
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
 
    if not image_paths:
        print("경고: 처리할 이미지가 없습니다.")
        return 0
 
    print(f"총 {len(image_paths)}개의 이미지 처리 시작...")
    print(f"병렬 처리 워커 수: {num_workers}")
    
    # Manager 제거: 단순 인자 전달로 변경
    # args_list에 enable_flops 정보를 포함
    args_list = [(img_path, SOURCE_DIRS, enable_flops) for img_path in image_paths]
    
    total_success = 0
    total_failed = 0
    total_operations = 0
    
    # 배치 단위 처리 루프
    # chunksize를 설정하여 imap 성능 최적화
    chunk_size = max(1, len(args_list) // (num_workers * 4))

    with Pool(processes=num_workers) as pool:
        # tqdm을 전체 진행률로 표시하기 위해 imap 사용
        results_iterator = pool.imap(process_single_image, args_list, chunksize=chunk_size)
        
        batch_buffer = []
        
        for result in tqdm(results_iterator, total=len(args_list), desc="Color Moment 추출 중"):
            batch_buffer.append(result)
            
            if len(batch_buffer) >= BATCH_SIZE:
                # FLOPS 집계
                if enable_flops:
                    total_operations += sum(r[1] for r in batch_buffer)
                
                # DB 저장
                success_count = save_to_database(batch_buffer, DB_PATH)
                total_success += success_count
                total_failed += (len(batch_buffer) - success_count)
                
                batch_buffer = [] # 버퍼 비우기

        # 남은 버퍼 처리
        if batch_buffer:
            if enable_flops:
                total_operations += sum(r[1] for r in batch_buffer)
            success_count = save_to_database(batch_buffer, DB_PATH)
            total_success += success_count
            total_failed += (len(batch_buffer) - success_count)

    print(f"\n=== RGB Color Moment 특징 추출 완료! ===")
    print(f"성공: {total_success}개, 실패: {total_failed}개")
    
    return total_operations

if __name__ == "__main__":
    mp.freeze_support()
    print("=== 병렬 처리 모드 (FLOPS 측정 포함 / 최적화 버전) ===")
    
    flops_counter = FLOPSCounter()
    
    import time
    start_time = time.time()
    
    total_ops = process_images_parallel(enable_flops=True)
    
    elapsed_time = time.time() - start_time
    
    if total_ops > 0:
        flops_counter.total_ops = total_ops
        flops_counter.num_images = 1 
        flops_result = flops_counter.get_result(elapsed_time)
        print(f"\n{'='*60}")
        print("FLOPS 측정 결과")
        print(f"{'='*60}")   
        print(f"총 연산 횟수:        {total_ops:,} FLOPs")
        print(f"소요 시간:          {elapsed_time:.2f}초")
        print(f"FLOPS:              {total_ops/elapsed_time:,.0f} ops/sec")
        print(f"GFLOPS:             {(total_ops/elapsed_time)/1e9:.4f} GFLOPS")
        print(f"{'='*60}")
    
    print("\n\n")
    print("===========Hello, this is Yubin Yang.=============================")
    print("=========== I am Korean.==========================================")