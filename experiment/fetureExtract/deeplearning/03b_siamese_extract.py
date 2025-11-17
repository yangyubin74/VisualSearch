import tensorflow as tf
import numpy as np
import common_config as cfg
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

def extract_features_batch(feature_model, image_paths, batch_size=32):
    """[최적화] 배치 단위 특징 추출 (Siamese용)"""
    num_images = len(image_paths)
    
    # 출력 차원 확인
    if hasattr(feature_model.output, 'shape'):
        feature_dim = feature_model.output.shape[-1]
    else:
        feature_dim = feature_model.output[0].shape[-1]
        
    features = np.zeros((num_images, feature_dim), dtype=np.float32)
    failed_indices = []
    target_size = cfg.IMG_SIZE_SIAMESE  # Siamese용 이미지 크기
    
    for i in tqdm(range(0, num_images, batch_size), desc="  배치 처리 중"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_indices = []
        
        for j, img_path in enumerate(batch_paths):
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                batch_images.append(img_array)
                batch_indices.append(i + j)
            except Exception as e:
                print(f"\n[경고] 로드 실패: {img_path}")
                failed_indices.append(i + j)
        
        if batch_images:
            try:
                batch_array = np.array(batch_images)
                # Siamese(EfficientNet Backbone) 전처리 적용
                batch_preprocessed = preprocess_input(batch_array)
                
                batch_features = feature_model.predict(batch_preprocessed, verbose=0)
                
                for local_idx, global_idx in enumerate(batch_indices):
                    features[global_idx] = batch_features[local_idx].flatten()
            except Exception as e:
                print(f"\n[오류] 예측 실패: {e}")
                failed_indices.extend(batch_indices)
                
    return features, failed_indices

def main():
    print("\n--- [Siamese Network] 특징 추출 시작 (배치 가속) ---")
    
    cfg.create_directories()
    image_paths, _ = cfg.load_image_paths()
    
    # 학습된 Base Network 로드
    base_network_save_path = cfg.MODEL_SAVE_DIR / "siamesenetwork" / cfg.SEED_DIR / "base_network_best.h5"
    if not base_network_save_path.exists():
        print(f"오류: 모델 파일 없음 ({base_network_save_path})")
        return
        
    loaded_base_network = tf.keras.models.load_model(base_network_save_path)
    print("모델 로드 완료.")
    
    total_processed = 0
    
    for c in cfg.SPLITS:
        for s in cfg.CLASSES:
            print(f"\nSiamese: '{c}' / '{s}' 처리 중...")
            current_paths = image_paths[c][s]
            
            if not current_paths:
                continue
                
            # [핵심] 배치 처리 함수 호출
            features, failed_indices = extract_features_batch(
                loaded_base_network, 
                current_paths, 
                batch_size=32
            )
            
            # 저장
            save_path = cfg.FEATURE_SAVE_DIR / "siamesenetwork" / cfg.SEED_DIR / f"{c}_{s}_features.npy"
            np.save(save_path, features)
            print(f"저장 완료: {save_path.name} {features.shape}")
            
            # JSON 저장
            json_path = save_path.with_suffix('.json')
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(current_paths, f, ensure_ascii=False, indent=4)
            except:
                pass

            total_processed += len(current_paths)
    
    print("\n" + "="*60)
    print(f"총 {total_processed}개 처리 완료")
    print("="*60)

if __name__ == "__main__":
    cfg.measure_process_time(main)