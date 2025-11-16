import tensorflow as tf
import numpy as np
import common_config as cfg
import json

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input  # 변경

def preprocess_mobilenet_extract(img_array):  # 이름 변경
    """MobileNetV3 특징 추출용 전처리 (학습과 동일)"""
    return preprocess_input(img_array)

def main():
    print("\n--- [MobileNetV3] 특징 추출 시작 ---")  # 변경
    
    # 1. 공통 설정에서 경로 및 파일 목록 로드
    cfg.create_directories()
    image_paths, _ = cfg.load_image_paths()
    
    # 2. 학습된 Base Network 로드
    # common_config.py의 create_directories에 "mobilenetv3" 추가 가정
    base_network_save_path = cfg.MODEL_SAVE_DIR / "mobilenetv3" /cfg.SEED_DIR/ "base_network_best.h5"  # 변경
    if not base_network_save_path.exists():
        print(f"오류: 학습된 Base Network 모델이 없습니다! ({base_network_save_path})")
        print("먼저 04a_mobilenet_train.py를 실행하세요.")  # 변경
        return
        
    loaded_base_network = tf.keras.models.load_model(base_network_save_path)
    
    # 특징 벡터 차원 안전하게 추출
    try:
        if hasattr(loaded_base_network.output, 'shape'):
            feature_dim = loaded_base_network.output.shape[-1]
        else:
            feature_dim = loaded_base_network.output[0].shape[-1]
        print(f"MobileNetV3 모델 로드 완료. 특징 벡터 차원: {feature_dim}")   
    except Exception as e:
        print(f"경고: 특징 벡터 차원을 가져올 수 없습니다: {e}")
    
    # 3. 모든 분할 및 클래스에 대해 특징 추출 실행
    total_processed = 0
    total_features_saved = 0
    
    IMG_SIZE_MOBILENET = cfg.IMG_SIZE_MOBILENET
    
    for c in cfg.SPLITS:
        for s in cfg.CLASSES:
            print(f"\nMobileNetV3: '{c}' / '{s}' 처리 중...")   
            
            current_paths = image_paths[c][s]
            if not current_paths:
                print(f"경고: '{c}' / '{s}'에 이미지가 없습니다.")
                continue
                
            # 공통 추출 함수 호출
            features = cfg.extract_features_from_paths(
                current_paths, 
                loaded_base_network, 
                preprocess_mobilenet_extract,  
                IMG_SIZE_MOBILENET,            
                desc=f"MobileNetV3 '{c}/{s}'"  
            )
            
            # 특징 추출 검증
            if features.size == 0:
                print(f"경고: '{c}' / '{s}'에서 추출된 특징이 없습니다!")
                continue
                
            if np.isnan(features).any():
                print(f"경고: '{c}' / '{s}' 특징에 NaN 값이 포함되어 있습니다!")
                features = np.nan_to_num(features, nan=0.0)
            
            # 4. 특징점 저장
            save_path = cfg.FEATURE_SAVE_DIR / "mobilenetv3"/ cfg.SEED_DIR / f"{c}_{s}_features.npy"  # 변경
            np.save(save_path, features)
            print(f"저장 완료: {save_path} (형태: {features.shape})")
            
            filename_save_path = save_path.with_suffix('.json')
            try:
                with open(filename_save_path, 'w', encoding='utf-8') as f:
                    json.dump(current_paths, f, ensure_ascii=False, indent=4)
                print(f"파일명 리스트 저장 완료: {filename_save_path.name}")
            except Exception as e:
                print(f"파일명 리스트 저장 실패: {e}")

            # 통계 업데이트
            total_processed += len(current_paths)
            total_features_saved += features.shape[0]
    
    # 최종 요약
    print("\n" + "="*60)
    print(f"총 {total_processed}개 이미지 처리 완료")
    print(f"총 {total_features_saved}개 특징 벡터 추출 및 저장 완료")
    print("="*60)
    print("\n--- [MobileNetV3] 모든 특징 추출 완료 ---")  # 변경

if __name__ == "__main__":
    
    cfg.measure_process_time(main)