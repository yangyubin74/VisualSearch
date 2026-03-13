import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dropout

import common_config as cfg
from common_utility import FLOPSCalculator
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 🔧 하이퍼파라미터 설정
# -----------------------------------------------------------
BATCH_SIZE = 32
MAX_EPOCHS = 100
FINE_TUNE_LAYERS = 10
LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

# -----------------------------------------------------------
# 학습 과정 시각화 함수
# -----------------------------------------------------------
def plot_history(history, save_path):
    """학습 히스토리를 시각화합니다 (Loss 및 Accuracy)."""
    plt.figure(figsize=(14, 5))

    # 1. Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"학습 그래프 저장 완료: {save_path}")
    plt.close()

# -----------------------------------------------------------
# 모델 생성 함수
# -----------------------------------------------------------
def create_model(num_classes, fine_tune_layers=FINE_TUNE_LAYERS):
    """EfficientNet 모델 생성 및 partial fine-tuning 설정"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*cfg.IMG_SIZE_EFFICIENTNET, 3)
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # 30% 드롭아웃
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Partial fine-tuning 설정
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True
    
    return model, base_model

# -----------------------------------------------------------
# 데이터 생성기 구성 함수
# -----------------------------------------------------------
def create_data_generators(train_dir, validation_dir, use_augmentation=True):
    """훈련 및 검증 데이터 생성기를 구성합니다."""
    
    # 훈련 데이터 생성기
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        print("✅ 데이터 증강(Augmentation) 활성화")
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
        print("ℹ️ 데이터 증강(Augmentation) 비활성화")
    
    # 검증 데이터 생성기 (증강 금지)
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    # 훈련 데이터 생성기
    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=cfg.IMG_SIZE_EFFICIENTNET,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    # 검증 데이터 생성기
    validation_generator = validation_datagen.flow_from_directory(
        directory=str(validation_dir),
        target_size=cfg.IMG_SIZE_EFFICIENTNET,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

# -----------------------------------------------------------
# 데이터셋 정보 출력 함수
# -----------------------------------------------------------
def print_dataset_info(train_generator, validation_generator):
    """데이터셋 정보를 출력."""
    num_classes = len(train_generator.class_indices)
    
    print(f"\n{'='*60}")
    print(f"데이터셋 정보")
    print(f"{'='*60}")
    print(f"훈련 이미지 개수: {train_generator.samples}")
    print(f"검증 이미지 개수: {validation_generator.samples}")
    print(f"클래스 개수: {num_classes}")
    print(f"클래스 목록: {list(train_generator.class_indices.keys())}")
    
    # 클래스별 샘플 수 출력
    print(f"클래스별 훈련 데이터 분포:")
    for class_name, class_idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        count = sum(train_generator.classes == class_idx)
        print(f"  - {class_name}: {count}개")
    print(f"{'='*60}\n")
    
    return num_classes

# -----------------------------------------------------------
# 데이터 검증 함수
# -----------------------------------------------------------
def validate_data(train_dir, validation_dir, train_generator):
    """데이터 경로 및 최소 요구사항을 검증."""
    
    # 디렉토리 존재 확인
    if not train_dir.exists():
        raise FileNotFoundError(f"훈련 데이터 경로를 찾을 수 없습니다: {train_dir}")
    if not validation_dir.exists():
        raise FileNotFoundError(f"검증 데이터 경로를 찾을 수 없습니다: {validation_dir}")
    
    # 최소 샘플 수 검증
    if train_generator.samples < 10:
        raise ValueError(f"훈련 데이터가 너무 적습니다: {train_generator.samples}개 (최소 10개 필요)")
    
    print("데이터 검증 완료")

# -----------------------------------------------------------
# 메인 학습 함수
# -----------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("[EfficientNet] Partial Fine-tuning 시작")
    print("="*60)

    # 1. 경로 생성
    cfg.create_directories()

    # 2. 데이터 경로 설정
    train_dir = cfg.BASE_IMAGE_DIR / "train"
    validation_dir = cfg.BASE_IMAGE_DIR / "validation"

    print(f"  데이터 경로:")
    print(f"  훈련: {train_dir}")
    print(f"  검증: {validation_dir}")

    # 3. 데이터 생성기 구성
    train_generator, validation_generator = create_data_generators(
        train_dir, 
        validation_dir,
        use_augmentation=False  
    )

    # 4. 데이터 검증
    validate_data(train_dir, validation_dir, train_generator)

    # 5. 데이터셋 정보 출력
    num_classes = print_dataset_info(train_generator, validation_generator)

    # 6. 모델 생성
    print("모델 생성 중...")
    model, base_model = create_model(num_classes, FINE_TUNE_LAYERS)
    
    # 7. 모델 정보 출력
    print(f"\n{'='*60}")
    print("모델 구성 정보")
    print(f"{'='*60}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = sum([tf.keras.backend.count_params(w) for w in model.weights])
    print(f"전체 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    print(f"학습 비율: {100*trainable_params/total_params:.1f}%")
    print(f"Fine-tuning 레이어 수: {FINE_TUNE_LAYERS}")

    # 7-1. FLOPS 측정
    flops_calculator = FLOPSCalculator()
    flops = flops_calculator.calculate(
        model, 
        input_shape=(1, *cfg.IMG_SIZE_EFFICIENTNET, 3), 
        model_name="EfficientNet",
        data_generator=train_generator
    )
    print(flops)
    print(f"{'='*60}\n")

    # 8. 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 9. 콜백 정의
    save_path = cfg.MODEL_SAVE_DIR / "efficientnet"/cfg.SEED_DIR / "efficientnet_best.h5"
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # ModelCheckpoint
    mc = ModelCheckpoint(
        str(save_path),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
    
    # ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks_list = [es, mc, reduce_lr]

    # 10. 모델 학습
    print("="*60)
    print("학습 시작")
    print("="*60)
    print(f"최대 Epoch: {MAX_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print("="*60 + "\n")
    
    history = model.fit(
        train_generator,
        epochs=MAX_EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks_list
    )

    # 11. 학습 결과 출력
    print("\n" + "="*60)
    print("[EfficientNet] Partial Fine-tuning 완료")
    print("="*60)
    print(f"최적 모델 저장 위치: {save_path}")
    
    # 최종 성능 출력
    best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    best_val_loss = min(history.history['val_loss'])
    best_val_acc = history.history['val_accuracy'][best_epoch - 1]
    
    print(f"   최종 성능:")
    print(f"  - Best Epoch: {best_epoch}")
    print(f"  - Best Validation Loss: {best_val_loss:.4f}")
    print(f"  - Best Validation Accuracy: {best_val_acc:.4f}")
    print("="*60 + "\n")

    # 12. 학습 과정 시각화
    plot_save_path = cfg.MODEL_SAVE_DIR / "efficientnet"/cfg.SEED_DIR / "efficientnet_loss_plot.png"
    plot_history(history, plot_save_path)


if __name__ == "__main__":
    
    cfg.measure_process_time(main)