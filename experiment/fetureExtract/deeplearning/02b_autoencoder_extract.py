import tensorflow as tf
import numpy as np
import common_config as cfg
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def extract_features_batch_ae(encoder_model, image_paths, batch_size=32):
    """[최적화] 배치 단위 특징 추출 (Autoencoder용)"""
    num_images = len(image_paths)
    feature_dim = encoder_model.output.shape[-1]
    features = np.zeros((num_images, feature_dim), dtype=np.float32)
    failed_indices = []
    target_size = cfg.IMG_SIZE_AE   
    
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
                
                # Autoencoder 전처리: 0~1 사이로 정규화 (float32 변환 포함)
                batch_preprocessed = batch_array.astype('float32') / 255.0
                
                # 예측
                batch_features = encoder_model.predict(batch_preprocessed, verbose=0)
                
                for local_idx, global_idx in enumerate(batch_indices):
                    features[global_idx] = batch_features[local_idx].flatten()
            except Exception as e:
                print(f"\n[오류] 예측 실패: {e}")
                failed_indices.extend(batch_indices)
                
    return features, failed_indices

def main():
    print("\n" + "="*60)
    print("  [Autoencoder] 특징 추출 시작 (배치 가속)")
    print("="*60)
    
    cfg.create_directories()
    image_paths, _ = cfg.load_image_paths()
    
    encoder_save_path = cfg.MODEL_SAVE_DIR / "autoencoder" / cfg.SEED_DIR / "encoder_model.h5"
    if not encoder_save_path.exists():
        print(f"오류: 모델 파일 없음")
        return
    
    encoder = tf.keras.models.load_model(encoder_save_path)
    print("모델 로드 완료.")
    
    total_processed = 0
    
    for c in cfg.SPLITS:
        for s in cfg.CLASSES:
            print(f"\n[클래스: {c} / {s}]")
            current_paths = image_paths[c][s]
            
            if not current_paths:
                continue
            
            # [핵심] 배치 처리 함수 호출
            features, failed_indices = extract_features_batch_ae(
                encoder, 
                current_paths, 
                batch_size=32
            )
            
            # 저장
            save_path = cfg.FEATURE_SAVE_DIR / "autoencoder" / cfg.SEED_DIR / f"{c}_{s}_features.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, features)
            print(f"  저장 완료: {save_path.name} {features.shape}")
            
            # JSON 저장
            json_path = save_path.with_suffix('.json')
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(current_paths, f, ensure_ascii=False, indent=4)
            except:
                pass

            total_processed += len(current_paths)
    
    print("\n" + "="*60)
    print(f"  [Autoencoder] 특징 추출 완료: {total_processed}개")
    print("="*60)

if __name__ == "__main__":
    cfg.measure_process_time(main)