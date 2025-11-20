"""
RGB 색공간 기반 Color Moment 특징 추출 및 DB 저장 (병렬처리 + FLOPS 측정)
"""
import cv2
import numpy as np
import sqlite3
import glob
import json
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Manager

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import config
from common_utility import (
    extract_color_moment_rgb, 
    measure_process_time, 
    get_category_from_path_dir,
    FLOPSCounter,
    measure_process_time_with_flops
)

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


def process_single_image(args, shared_counter=None):
    """
    단일 이미지 처리 + FLOPS 카운팅
    
    Args:
        args: (image_path, source_dirs) 튜플
        shared_counter: 멀티프로세싱용 공유 카운터 (선택)
    """
    image_path, source_dirs = args
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, 0
        
        # FLOPS 카운트 (각 프로세스에서 개별 계산)
        ops_count = 0
        if shared_counter is not None:
            counter = FLOPSCounter()
            ops_count = counter.count_color_moment_ops(image.shape)
        
        features = extract_color_moment_rgb(image)
        features_json = json.dumps(features.tolist())
        category = get_category_from_path_dir(image_path, source_dirs)
        
        return (image_path, category, features_json), ops_count
    
    except Exception as e:
        print(f"\n에러: {image_path} 처리 중 오류 발생 - {e}")
        return None, 0


def save_to_database(results, db_path):
    """처리 결과를 데이터베이스에 저장"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # None이 아닌 결과만 필터링 (결과는 (data, ops_count) 튜플)
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
    """
    병렬 처리로 이미지 특징 추출
    
    Args:
        num_workers: 워커 프로세스 수 (None이면 자동 설정)
        enable_flops: FLOPS 측정 활성화 여부
    
    Returns:
        total_operations: 총 연산 횟수 (FLOPS 측정 시)
    """
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
        return 0
 
    print(f"총 {len(image_paths)}개의 이미지 처리 시작...")
    print(f"병렬 처리 워커 수: {num_workers}")
    
    if enable_flops:
        print(f"FLOPS 측정: 활성화")
    
    # 병렬처리를 위한 인자 준비
    args_list = [(img_path, SOURCE_DIRS) for img_path in image_paths]
    
    # 공유 카운터 설정 (멀티프로세싱용)
    manager = Manager()
    shared_counter = manager.Namespace() if enable_flops else None
    
    # 배치 크기 설정
    batch_size = 1000
    total_success = 0
    total_failed = 0
    total_operations = 0
    
    # 배치 단위로 처리
    for i in range(0, len(args_list), batch_size):
        batch_args = args_list[i:i + batch_size]
        
        # 병렬처리로 특징 추출
        with Pool(processes=num_workers) as pool:
            # partial을 사용하여 shared_counter 전달
            from functools import partial
            process_func = partial(process_single_image, shared_counter=shared_counter)
            
            results = list(tqdm(
                pool.imap(process_func, batch_args),
                total=len(batch_args),
                desc=f"배치 {i//batch_size + 1}/{(len(args_list)-1)//batch_size + 1} 처리 중"
            ))
        
        # FLOPS 카운트 누적
        if enable_flops:
            batch_ops = sum(r[1] for r in results)
            total_operations += batch_ops
        
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
    
    return total_operations


if __name__ == "__main__":
    # Windows에서 multiprocessing 사용 시 필요
    mp.freeze_support()
    
    print("=== 병렬 처리 모드 (FLOPS 측정 포함) ===")
    
    # FLOPS 카운터 생성
    flops_counter = FLOPSCounter()
    
    # 시간 측정 시작
    import time
    start_time = time.time()
    
    # 병렬 처리 실행 및 총 연산 횟수 반환
    total_ops = process_images_parallel(enable_flops=True)
    
    # 실행 시간 계산
    elapsed_time = time.time() - start_time
    
    # FLOPS 결과 생성 및 출력
    if total_ops > 0:
        flops_counter.total_ops = total_ops
        flops_counter.num_images = 1  # 임시값 (실제로는 process_images_parallel에서 반환)
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