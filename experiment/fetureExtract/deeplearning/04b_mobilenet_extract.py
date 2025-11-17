import tensorflow as tf
import numpy as np
import common_config as cfg
import json
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# MobileNetV3 전용 전처리 함수 임포트
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def extract_features_batch(feature_model, image_paths, batch_size=32):
    """
    [최적화 적용] 배치 단위로 이미지를 묶어서 특징을 추출합니다.
    속도 향상의 핵심입니다.
    """
    num_images = len(image_paths)
    # 모델의 출력 차원 확인 (MobileNetV3는 보통 1280 또는 1024)
    if hasattr(feature_model.output, 'shape'):
        feature_dim = feature_model.output.shape[-1]
    else:
        feature_dim = feature_model.output[0].shape[-1]
        
    features = np.zeros((num_images, feature_dim), dtype=np.float32)
    failed_indices = []
    
    # IMG_SIZE는 MobileNet 설정 사용
    target_size = cfg.IMG_SIZE_MOBILENET
    
    for i in tqdm(range(0, num_images, batch_size), desc="  배치 처리 중"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_indices = []
        
        # 1. 배치 이미지 로드 및 전처리
        for j, img_path in enumerate(batch_paths):
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                batch_images.append(img_array)
                batch_indices.append(i + j)
            except Exception as e:
                print(f"\n[경고] 이미지 로드 실패: {img_path} -> {e}")
                failed_indices.append(i + j)
        
        # 2. 배치 예측 (GPU 가속 활용)
        if batch_images:
            try:
                batch_array = np.array(batch_images)
                # MobileNetV3 전용 전처리 적용
                batch_preprocessed = preprocess_input(batch_array)
                
                # verbose=0으로 설정하여 로그 출력 부하 감소
                batch_features = feature_model.predict(batch_preprocessed, verbose=0)
                
                # 결과 저장
                for local_idx, global_idx in enumerate(batch_indices):
                    features[global_idx] = batch_features[local_idx].flatten()
                    
            except Exception as e:
                print(f"\n[오류] 배치 예측 실패 -> {e}")
                failed_indices.extend(batch_indices)
    
    return features, failed_indices

def main():
    print("\n" + "="*60)
    print(" [MobileNetV3] 특징 추출 시작 (배치 가속 모드)")
    print("="*60)
    
    # 1. 공통 설정 로드
    cfg.create_directories()
    image_paths, _ = cfg.load_image_paths()
    
    # 2. 학습된 Base Network 로드
    base_network_save_path = cfg.MODEL_SAVE_DIR / "mobilenetv3" / cfg.SEED_DIR / "base_network_best.h5"
    
    if not base_network_save_path.exists():
        print(f"오류: 모델 파일이 없습니다! ({base_network_save_path})")
        return
        
    print(f"모델 로드 중: {base_network_save_path.name}...")
    loaded_base_network = tf.keras.models.load_model(base_network_save_path)
    print("모델 로드 완료.")
    
    # 3. 특징 추출 실행
    total_processed = 0
    total_failed = 0
    
    for c in cfg.SPLITS:
        for s in cfg.CLASSES:
            print(f"\n{'─'*60}")
            print(f"처리 중: MobileNetV3 / {c} / {s}")
            
            current_paths = image_paths[c][s]
            if not current_paths:
                print("경고: 이미지가 없습니다.")
                continue
                
            # [핵심 변경] 배치 처리 함수 사용 (batch_size=32)
            features, failed_indices = extract_features_batch(
                loaded_base_network, 
                current_paths, 
                batch_size=32
            )
            
            # 유효성 검증 (NaN 체크)
            if np.isnan(features).any():
                print("경고: NaN 값이 발견되어 0으로 대체합니다.")
                features = np.nan_to_num(features, nan=0.0)

            # 4. 저장
            save_dir = cfg.FEATURE_SAVE_DIR / "mobilenetv3" / cfg.SEED_DIR
            save_dir.mkdir(parents=True, exist_ok=True) # 폴더 안전 생성
            
            npy_path = save_dir / f"{c}_{s}_features.npy"
            json_path = save_dir / f"{c}_{s}_features.json"
            
            # .npy 저장
            np.save(npy_path, features)
            print(f"  -> 특징 벡터 저장 완료: {npy_path.name} {features.shape}")
            
            # .json 저장
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(current_paths, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"  -> 파일명 리스트 저장 실패: {e}")

            total_processed += len(current_paths)
            total_failed += len(failed_indices)

    print("\n" + "="*60)
    print(f" 최종 완료: 총 {total_processed}개 처리 (실패 {total_failed}개)")
    print("="*60)

if __name__ == "__main__":
    cfg.measure_process_time(main)