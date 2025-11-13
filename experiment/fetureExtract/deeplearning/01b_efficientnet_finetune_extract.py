import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from tqdm import tqdm
import os
import common_config as cfg
import json

def extract_features_batch(feature_model, image_paths, batch_size=32):
    """배치 단위로 특징 추출"""
    num_images = len(image_paths)
    feature_dim = feature_model.output.shape[-1]
    features = np.zeros((num_images, feature_dim), dtype=np.float32)
    failed_indices = []
    
    for i in tqdm(range(0, num_images, batch_size), desc="배치 처리"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_indices = []
        
        # 배치 이미지 로드
        for j, img_path in enumerate(batch_paths):
            try:
                img = load_img(img_path, target_size=cfg.IMG_SIZE_EFFICIENTNET)
                img_array = img_to_array(img)
                batch_images.append(img_array)
                batch_indices.append(i + j)
            except Exception as e:
                print(f"\n경고: {img_path} 로드 실패 → {e}")
                failed_indices.append(i + j)
        
        # 배치 예측
        if batch_images:
            try:
                batch_array = np.array(batch_images)
                batch_preprocessed = preprocess_input(batch_array)
                batch_features = feature_model.predict(batch_preprocessed, verbose=0)
                
                # 결과 저장
                for local_idx, global_idx in enumerate(batch_indices):
                    features[global_idx] = batch_features[local_idx].flatten()
            except Exception as e:
                print(f"\n오류: 배치 예측 실패 → {e}")
                failed_indices.extend(batch_indices)
    
    return features, failed_indices


def validate_features(features, expected_count, class_name, split_name):
    """특징 벡터 유효성 검증"""
    issues = []
    
    # 개수 확인
    if features.shape[0] != expected_count:
        issues.append(f"개수 불일치: 예상 {expected_count}, 실제 {features.shape[0]}")
    
    # NaN/Inf 확인
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    if nan_count > 0:
        issues.append(f"NaN 값 {nan_count}개 발견")
    if inf_count > 0:
        issues.append(f"Inf 값 {inf_count}개 발견")
    
    # 제로 벡터 확인
    zero_vectors = np.sum(np.all(features == 0, axis=1))
    if zero_vectors > 0:
        issues.append(f"제로 벡터 {zero_vectors}개 발견")
    
    if issues:
        print(f" 검증 경고 ({class_name}/{split_name}):")
        for issue in issues:
            print(f"     - {issue}")
        return False
    
    return True


def main():
    print("\n" + "="*60)
    print("  EfficientNet 특징 추출 (파인튜닝 모델 기반)")
    print("="*60)
    
    # 1. 경로 및 이미지 목록 로드
    cfg.create_directories()
    image_paths, _ = cfg.load_image_paths()
    
    # 2. 파인튜닝된 EfficientNet 모델 로드
    model_path = cfg.MODEL_SAVE_DIR / "efficientnet" / "efficientnet_best.h5"
    if not model_path.exists():
        print(f"\n 오류: 파인튜닝된 모델이 존재하지 않습니다")
        print(f"   경로: {model_path}")
        print(f"   먼저 efficientnet_best.py를 실행하세요.")
        return
    
    print(f"\n 모델 로드 중...")
    full_model = load_model(model_path)
    print(f"   로드 완료: {model_path.name}")
    
    # 3. 특징 추출용 모델 구성 (마지막 Dense 레이어 제거)
    feature_model = Model(
        inputs=full_model.input, 
        outputs=full_model.layers[-2].output
    )
    feature_dim = feature_model.output.shape[-1]
    print(f"  특징 벡터 차원: {feature_dim}")
    
    # 4. 특징 추출 및 저장
    total_stats = {
        'processed': 0,
        'failed': 0,
        'validation_passed': 0,
        'validation_failed': 0
    }
    
    for c in cfg.SPLITS:
        for s in cfg.CLASSES:
            print(f"\n{'─'*60}")
            print(f"처리 중: {c} / {s}")
            print(f"{'─'*60}")
            
            current_paths = image_paths[c][s]
            if not current_paths:
                print(f"건너뜀: 이미지가 없습니다")
                continue
            
            print(f"   이미지 수: {len(current_paths)}")
            
            # 특징 추출
            features, failed_indices = extract_features_batch(
                feature_model, 
                current_paths,
                batch_size=32
            )
            
            # 통계 업데이트
            total_stats['processed'] += len(current_paths)
            total_stats['failed'] += len(failed_indices)
            
            # 실패한 이미지 정보 출력
            if failed_indices:
                print(f"\n   {len(failed_indices)}개 이미지 처리 실패:")
                for idx in failed_indices[:5]:  # 최대 5개만 표시
                    print(f"      - {current_paths[idx]}")
                if len(failed_indices) > 5:
                    print(f"      ... 외 {len(failed_indices)-5}개")
            
            # 유효성 검증
            is_valid = validate_features(features, len(current_paths), c, s)
            if is_valid:
                total_stats['validation_passed'] += 1
                print(f"   검증 통과")
            else:
                total_stats['validation_failed'] += 1
            
            # 저장
            save_path = cfg.FEATURE_SAVE_DIR / "efficientnet" / f"{c}_{s}_features.npy"
            np.save(save_path, features)
            print(f"   저장 완료: {save_path.name}")
            print(f"   형태: {features.shape}, 타입: {features.dtype}")

            # .npy 특징 벡터 저장
            save_path = cfg.FEATURE_SAVE_DIR / "efficientnet" / f"{c}_{s}_features.npy"
            np.save(save_path, features)
            print(f"   특징 벡터 저장 완료: {save_path.name}")
            print(f"   형태: {features.shape}, 타입: {features.dtype}")

              
          
              # .json 파일명 리스트 저장
              # (예: "train_shirt_features.npy" -> "train_shirt_features.json")
            filename_save_path = save_path.with_suffix('.json')
            try:
                with open(filename_save_path, 'w', encoding='utf-8') as f:
                    json.dump(current_paths, f, ensure_ascii=False, indent=4)
                print(f"   파일명 리스트 저장 완료: {filename_save_path.name}")
            except Exception as e:
                print(f"   파일명 리스트 저장 실패: {e}")
          
          
    
    # 5. 최종 통계 출력
    print(f"\n{'='*60}")
    print("  최종 통계")
    print(f"{'='*60}")
    print(f"   총 처리 이미지:     {total_stats['processed']:,}개")
    print(f"   처리 실패:          {total_stats['failed']:,}개 "
          f"({100*total_stats['failed']/max(total_stats['processed'],1):.2f}%)")
    print(f"   검증 통과:          {total_stats['validation_passed']}개 세트")
    print(f"   검증 실패:          {total_stats['validation_failed']}개 세트")
    print(f"\n 모든 특징 추출 완료!\n")


if __name__ == "__main__":
    cfg.measure_process_time(main)