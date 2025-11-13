
import tensorflow as tf
import numpy as np
import common_config as cfg
from pathlib import Path
import json   

def preprocess_ae_extract(img_array):
    """AE 특징 추출용 전처리 (학습과 동일)"""
    return img_array.astype('float32') / 255.0

def validate_features(features, name="features"):
    """특징 벡터 유효성 검증"""
    if np.isnan(features).any():
        print(f"경고: {name}에 NaN 값이 있습니다!")
        return False
    if np.isinf(features).any():
        print(f"경고: {name}에 Inf 값이 있습니다!")
        return False
    return True

def load_encoder_model(model_path):
    """인코더 모델 안전하게 로드"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f" 모델 로드 완료: {model_path}")
        print(f"  - 입력 크기: {model.input.shape}")
        print(f"  - 특징 벡터 차원: {model.output.shape[-1]}")
        return model
    except Exception as e:
        print(f" 모델 로드 실패: {e}")
        return None

def extract_and_save_features(encoder, image_paths, save_path, class_name, split_name):
    """특징 추출 및 저장 (검증 포함)"""
    if not image_paths:
        print(f"  '{class_name}/{split_name}': 이미지 없음")
        return False
    
    print(f"  → '{class_name}/{split_name}': {len(image_paths)}개 이미지 처리 중...")
    
    try:
        # 특징 추출
        features = cfg.extract_features_from_paths(
            image_paths, 
            encoder, 
            preprocess_ae_extract,
            cfg.IMG_SIZE_AE,
            desc=f"AE {class_name}/{split_name}"
        )
        
        # 유효성 검증
        if not validate_features(features, f"{class_name}/{split_name}"):
            print(f" 특징 벡터 검증 실패")
            return False
        
        # 저장
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, features)
        
        print(f"  저장 완료: {save_path.name}")
        print(f"  형태: {features.shape}, 범위: [{features.min():.3f}, {features.max():.3f}]")
        
        filename_save_path = save_path.with_suffix('.json')
        try:
            with open(filename_save_path, 'w', encoding='utf-8') as f:
                json.dump(image_paths, f, ensure_ascii=False, indent=4)
            print(f"  파일명 리스트 저장 완료: {filename_save_path.name}")
        except Exception as e:
            print(f"  파일명 리스트 저장 실패: {e}")


        # 메모리 해제
        del features
        
        return True
        
    except Exception as e:
        print(f"  오류 발생: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  [Autoencoder] 특징 추출 시작")
    print("="*60)
    
    # 1. 디렉토리 및 이미지 경로 준비
    cfg.create_directories()
    image_paths, _ = cfg.load_image_paths()
    
    # 2. 인코더 모델 로드
    encoder_save_path = cfg.MODEL_SAVE_DIR / "autoencoder" / "encoder_model.h5"
    if not encoder_save_path.exists():
        print(f"\n 오류: 학습된 인코더 모델이 없습니다!")
        print(f"  경로: {encoder_save_path}")
        print(f"  → 먼저 02a_autoencoder_train.py를 실행하세요.")
        return
    
    encoder = load_encoder_model(encoder_save_path)
    if encoder is None:
        return
    
    # 3. 특징 추출 통계
    total_processed = 0
    total_failed = 0
    
    # 4. 모든 클래스 및 분할에 대해 특징 추출
    for class_name in cfg.SPLITS:
        print(f"\n[클래스: {class_name}]")
        
        for split_name in cfg.CLASSES:
            current_paths = image_paths[class_name][split_name]
            save_path = cfg.FEATURE_SAVE_DIR / "autoencoder" / f"{class_name}_{split_name}_features.npy"
            
            success = extract_and_save_features(
                encoder, 
                current_paths, 
                save_path, 
                class_name, 
                split_name
            )
            
            if success:
                total_processed += len(current_paths)
            else:
                total_failed += 1
    
    # 5. 최종 결과 출력
    print("\n" + "="*60)
    print("  [Autoencoder] 특징 추출 완료")
    print("="*60)
    print(f" 처리된 이미지: {total_processed}개")
    if total_failed > 0:
        print(f" 실패한 작업: {total_failed}개")
    print(f" 저장 위치: {cfg.FEATURE_SAVE_DIR / 'autoencoder'}")
    print()

if __name__ == "__main__":
    cfg.measure_process_time(main)