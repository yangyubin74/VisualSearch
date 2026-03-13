import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt 
import numpy as np

from pathlib import Path

import common_config as cfg
from common_utility import FLOPSCalculator

def plot_ae_history(history, save_path):
    """Autoencoder의 학습 히스토리를 시각화하여 파일로 저장."""
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_path))  
    print(f"학습 그래프 저장 완료: {save_path}")
    plt.close()



def load_and_preprocess_ae(path):
    """AE 학습용 데이터 로더"""
    img = tf.io.read_file(path)
 
    img = tf.io.decode_image(img, channels=3, expand_animations=False) 
    img = tf.image.resize(img, [cfg.IMG_SIZE_AE[0], cfg.IMG_SIZE_AE[1]])
    img = tf.cast(img, tf.float32) / 255.0 # 0~1 정규화
    return img, img # Autoencoder는 입력과 타겟(정답)이 동일

def build_autoencoder(input_shape, latent_dim):
    """
    Autoencoder 모델 정의
    [중요] TF_DETERMINISTIC_OPS=1과 호환되도록 UpSampling2D 제거
    """
    IMG_H, IMG_W, _ = input_shape
    
    # Encoder
    input_img_ae = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img_ae)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # 잠재 벡터로 압축
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x_flat = Flatten()(x)
    encoded = Dense(latent_dim, activation='relu', name='latent_vector')(x_flat)
    
    # Decoder
    x = Dense(np.prod(shape_before_flatten), activation='relu')(encoded)
    x = Reshape(shape_before_flatten)(x)
    
    # [수정] Conv2DTranspose의 strides로 업샘플링 (Deterministic 호환)
    # UpSampling2D 대신 strides=(2,2) 사용
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), 
                       activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), 
                       activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), 
                       activation='relu', padding='same')(x)
    
    # 최종 출력
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', 
                             padding='same')(x)
    
    autoencoder = Model(input_img_ae, decoded)
    return autoencoder

def main():
    print("\n--- [Autoencoder] 학습 시작 ---")
    
    # 1. [수정] 경로를 EfficientNet과 동일한 방식으로 직접 설정
    cfg.create_directories()
    
    # EfficientNet과 동일하게 경로 정의
    train_dir = cfg.BASE_IMAGE_DIR / "train"
    validation_dir = cfg.BASE_IMAGE_DIR / "validation"

    print(f"  데이터 경로:")
    print(f"  훈련: {train_dir}")
    print(f"  검증: {validation_dir}")

    
    extensions = cfg.IMAGE_EXTENSIONS
    
    all_train_paths = []
    for ext in extensions:
        # rglob('**/')는 하위 폴더(dress, pants, shirt)를 모두 재귀적으로 검색
        all_train_paths.extend(train_dir.rglob(f'**/{ext}')) 

    all_validation_paths = []
    for ext in extensions:
        all_validation_paths.extend(validation_dir.rglob(f'**/{ext}'))

    # pathlib 객체를 문자열로 변환 (tf.data.Dataset은 문자열을 선호)
    all_train_paths = [str(p) for p in all_train_paths]
    all_validation_paths = [str(p) for p in all_validation_paths]

    # 1-2. 오류 검사 (기존 로직)
    if not all_train_paths or not all_validation_paths:
        print("오류: 학습 또는 테스트 이미지가 없습니다.")
        print(f"  -> Train 경로: {train_dir} (파일 {len(all_train_paths)}개 찾음)")
        print(f"  -> validation 경로: {validation_dir} (파일 {len(all_validation_paths)}개 찾음)")
        print("경로와 확장자(jpg, png 등)를 확인하세요.")
        return

    BATCH_SIZE = 32

    # 2. [수정] tf.data 파이프라인 생성 (훈련/검증 분리)
    # 훈련 데이터셋
    train_dataset_ae = tf.data.Dataset.from_tensor_slices(all_train_paths)
    train_dataset_ae = train_dataset_ae.shuffle(buffer_size=len(all_train_paths))
    train_dataset_ae = train_dataset_ae.map(load_and_preprocess_ae, 
                                            num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset_ae = train_dataset_ae.batch(BATCH_SIZE)
    train_dataset_ae = train_dataset_ae.repeat() # <--- [수정] 데이터셋이 무한 반복되도록 설정
    train_dataset_ae = train_dataset_ae.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # 검증 데이터셋
    validation_dataset_ae = tf.data.Dataset.from_tensor_slices(all_validation_paths)
    validation_dataset_ae = validation_dataset_ae.map(load_and_preprocess_ae,
                                                     num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset_ae = validation_dataset_ae.batch(BATCH_SIZE)
    validation_dataset_ae = validation_dataset_ae.repeat() # <--- [수정] 데이터셋이 무한 반복되도록 설정
    validation_dataset_ae = validation_dataset_ae.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Autoencoder 학습 데이터셋 준비 완료. (이미지 크기: {cfg.IMG_SIZE_AE})")
    print(f"훈련 이미지: {len(all_train_paths)}, 검증 이미지: {len(all_validation_paths)}")

    # 3. 모델 빌드 (기존과 동일)
    LATENT_DIM = 256
    autoencoder = build_autoencoder(
        (*cfg.IMG_SIZE_AE, 3), 
        LATENT_DIM
    )
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # 4. 콜백 정의 (기존과 동일)
    autoencoder_save_path = cfg.MODEL_SAVE_DIR / "autoencoder"/cfg.SEED_DIR / "autoencoder_best.h5"
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(str(autoencoder_save_path), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks_list = [es, mc]

    # 5. 학습 (기존과 동일)
    EPOCHS = 100 
    STEPS = len(all_train_paths) // BATCH_SIZE
    VALIDATION_STEPS = len(all_validation_paths) // BATCH_SIZE

    history = autoencoder.fit(
        train_dataset_ae, 
        epochs=EPOCHS, 
        steps_per_epoch=STEPS,
        validation_data=validation_dataset_ae, 
        validation_steps=VALIDATION_STEPS,     
        callbacks=callbacks_list               
    )


    # 6. 최적 Encoder 모델 저장 (기존과 동일)
    print(f"\n--- [Autoencoder] 학습 완료 ---")
    print(f"가장 좋았던 Autoencoder 로드 중... ({autoencoder_save_path})")


     # 7-1. FLOPS 측정
    flops_calculator = FLOPSCalculator()
    flops = flops_calculator.calculate_with_dataset(
    autoencoder,                               
    input_shape=(1, *cfg.IMG_SIZE_AE, 3),       
    model_name="Autoencoder",                  
    dataset=train_dataset_ae,                  
    num_samples=len(all_train_paths),          
    num_batches=STEPS                          
    )
    print(flops)
    print(f"{'='*60}\n")


    try:
        best_autoencoder = load_model(autoencoder_save_path)
        encoder_output = best_autoencoder.get_layer('latent_vector').output
        best_encoder = Model(best_autoencoder.input, encoder_output)
        
        encoder_save_path = cfg.MODEL_SAVE_DIR / "autoencoder"/cfg.SEED_DIR / "encoder_model.h5"
        best_encoder.save(encoder_save_path)
        
        print(f"최적의 Encoder 모델 추출 및 저장 완료: {encoder_save_path}")

    except Exception as e:
        print(f"최적 모델 로드 또는 Encoder 추출 실패: {e}")
        # ( ... 실패 시 백업 로직 ...)

    # 7. 학습 과정 시각화 (파일로 저장)
    plot_save_path = cfg.MODEL_SAVE_DIR / "autoencoder"/cfg.SEED_DIR / "autoencoder_loss_plot.png"
    plot_ae_history(history, plot_save_path)

if __name__ == "__main__":
    cfg.measure_process_time(main)