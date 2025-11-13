### ===================================================
### 02a. Autoencoder 모델 학습 (콜백 및 검증 추가)
### ===================================================

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt 
import numpy as np

# [추가] pathlib의 Path 객체 임포트 (rglob을 사용하기 위함)
from pathlib import Path

import common_config as cfg

# ... (plot_ae_history, load_and_preprocess_ae, build_autoencoder 함수는 동일) ...

# -----------------------------------------------------------
# [수정] 📊 오토인코더 학습 과정 시각화 (파일 저장용)
# -----------------------------------------------------------
def plot_ae_history(history, save_path):
    """Autoencoder의 학습 히스토리를 시각화하여 파일로 저장합니다."""
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_path)) # Path 객체일 수 있으므로 str()로 변환
    print(f"학습 그래프 저장 완료: {save_path}")
    plt.close()

# ... (load_and_preprocess_ae, build_autoencoder 함수는 동일) ...

def load_and_preprocess_ae(path):
    """AE 학습용 데이터 로더"""
    img = tf.io.read_file(path)
    # [수정] expand_animations=False 추가 (GIF 등 처리)
    img = tf.io.decode_image(img, channels=3, expand_animations=False) 
    img = tf.image.resize(img, [cfg.IMG_SIZE_AE[0], cfg.IMG_SIZE_AE[1]])
    img = tf.cast(img, tf.float32) / 255.0 # 0~1 정규화
    return img, img # Autoencoder는 입력과 타겟(정답)이 동일
def build_autoencoder(input_shape, latent_dim):
    """Autoencoder 모델 정의"""
    IMG_H, IMG_W, _ = input_shape
    
    # Encoder
    input_img_ae = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img_ae)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x) # (입력 128x128 기준 -> 16x16x128)
    
    # 잠재 벡터(Latent Vector)로 압축
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x_flat = Flatten()(x)
    # 'latent_vector' 이름은 나중에 Encoder 모델을 추출할 때 사용됩니다.
    encoded = Dense(latent_dim, activation='relu', name='latent_vector')(x_flat)
    
    # Decoder
    # 잠재 벡터를 다시 2D 특징 맵 형태로 복원
    x = Dense(np.prod(shape_before_flatten), activation='relu')(encoded)
    x = Reshape(shape_before_flatten)(x)
    
    # Encoder의 역순으로 이미지 복원 (Conv2DTranspose + UpSampling2D)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # 최종 원본 이미지 채널(3)과 크기로 복원
    # 픽셀 값 0~1 복원을 위해 'sigmoid' 활성화 함수 사용
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    # 전체 Autoencoder 모델
    autoencoder = Model(input_img_ae, decoded)
    
    # autoencoder 모델만 반환 (Encoder는 학습 후 추출)
    return autoencoder

def main():
    print("\n--- [Autoencoder] 학습 시작 ---")
    
    # 1. [수정] 경로를 EfficientNet과 동일한 방식으로 직접 설정
    cfg.create_directories()
    
    # EfficientNet과 동일하게 경로 정의
    train_dir = cfg.BASE_IMAGE_DIR / "train"
    validation_dir = cfg.BASE_IMAGE_DIR / "test"

    print(f"Train (AE) 경로 스캔: {train_dir}")
    print(f"Test (AE) 경로 스캔: {validation_dir}")

    # [수정] cfg.load_image_paths() 대신 rglob으로 파일 목록 직접 스캔
    # 지원할 이미지 확장자 (필요시 .png 등 추가)
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    
    all_train_paths = []
    for ext in extensions:
        # rglob('**/')는 하위 폴더(dress, pants, shirt)를 모두 재귀적으로 검색
        all_train_paths.extend(train_dir.rglob(f'**/{ext}')) 

    all_test_paths = []
    for ext in extensions:
        all_test_paths.extend(validation_dir.rglob(f'**/{ext}'))

    # [수정] pathlib 객체를 문자열로 변환 (tf.data.Dataset은 문자열을 선호)
    all_train_paths = [str(p) for p in all_train_paths]
    all_test_paths = [str(p) for p in all_test_paths]

    # 1-2. 오류 검사 (기존 로직)
    if not all_train_paths or not all_test_paths:
        print("오류: 학습 또는 테스트 이미지가 없습니다.")
        print(f"  -> Train 경로: {train_dir} (파일 {len(all_train_paths)}개 찾음)")
        print(f"  -> Test 경로: {validation_dir} (파일 {len(all_test_paths)}개 찾음)")
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
    validation_dataset_ae = tf.data.Dataset.from_tensor_slices(all_test_paths)
    validation_dataset_ae = validation_dataset_ae.map(load_and_preprocess_ae,
                                                     num_parallel_calls=tf.data.AUTOTUNE)
    validation_dataset_ae = validation_dataset_ae.batch(BATCH_SIZE)
    validation_dataset_ae = validation_dataset_ae.repeat() # <--- [수정] 데이터셋이 무한 반복되도록 설정
    validation_dataset_ae = validation_dataset_ae.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Autoencoder 학습 데이터셋 준비 완료. (이미지 크기: {cfg.IMG_SIZE_AE})")
    print(f"훈련 이미지: {len(all_train_paths)}, 검증 이미지: {len(all_test_paths)}")

    # 3. 모델 빌드 (기존과 동일)
    LATENT_DIM = 256
    autoencoder = build_autoencoder(
        (*cfg.IMG_SIZE_AE, 3), 
        LATENT_DIM
    )
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    # 4. 콜백 정의 (기존과 동일)
    autoencoder_save_path = cfg.MODEL_SAVE_DIR / "autoencoder" / "autoencoder_best.h5"
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(str(autoencoder_save_path), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks_list = [es, mc]

    # 5. 학습 (기존과 동일)
    EPOCHS = 100 
    STEPS = len(all_train_paths) // BATCH_SIZE
    VALIDATION_STEPS = len(all_test_paths) // BATCH_SIZE

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

    try:
        best_autoencoder = load_model(autoencoder_save_path)
        encoder_output = best_autoencoder.get_layer('latent_vector').output
        best_encoder = Model(best_autoencoder.input, encoder_output)
        
        encoder_save_path = cfg.MODEL_SAVE_DIR / "autoencoder" / "encoder_model.h5"
        best_encoder.save(encoder_save_path)
        
        print(f"최적의 Encoder 모델 추출 및 저장 완료: {encoder_save_path}")

    except Exception as e:
        print(f"최적 모델 로드 또는 Encoder 추출 실패: {e}")
        # ( ... 실패 시 백업 로직 ...)

    # 7. 학습 과정 시각화 (파일로 저장)
    plot_save_path = cfg.MODEL_SAVE_DIR / "autoencoder" / "autoencoder_loss_plot.png"
    plot_ae_history(history, plot_save_path)

if __name__ == "__main__":
    cfg.measure_process_time(main)