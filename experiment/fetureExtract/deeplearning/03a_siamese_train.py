
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import mixed_precision, regularizers
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random

import common_config as cfg

# --- 시각화 함수 ---
def plot_siamese_history(history, save_path):
    """Siamese Network의 학습 히스토리를 시각화하여 파일로 저장."""
    plt.figure(figsize=(7, 5))
    
    plt.plot(history.history['triplet_loss'], label='Train Loss')
    if 'val_triplet_loss' in history.history:
        plt.plot(history.history['val_triplet_loss'], label='Validation Loss')
    
    plt.title('Siamese (Triplet) Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Triplet)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_path))
    print(f"학습 그래프 저장 완료: {save_path}")
    plt.close()


# --- 커스텀 모델 저장 콜백 ---
class SaveBaseNetworkCallback(Callback):
    """
    Epoch 종료 시, 'val_triplet_loss'를 모니터링하여
    가장 성능이 좋았던 시점의 'base_network' 모델만 저장.
    """
    def __init__(self, save_path, monitor='val_triplet_loss', mode='min'):
        super().__init__()
        self.save_path = str(save_path)
        self.monitor = monitor
        self.mode = mode
        self.best = np.Inf if mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            print(f"\n Warning: {self.monitor} not found in logs")
            return

        improved = (self.mode == 'min' and current < self.best) or \
                   (self.mode == 'max' and current > self.best)
        
        if improved:
            print(f"\n Epoch {epoch+1}: {self.monitor} improved {self.best:.5f} → {current:.5f}")
            self.best = current
            try:
                self.model.base_network.save(self.save_path)
                print(f"  Model saved: {self.save_path}")
            except Exception as e:
                print(f"  Save error: {e}")
        else:
            print(f"\n- Epoch {epoch+1}: {self.monitor} = {current:.5f} (best: {self.best:.5f})")


# --- 데이터 증강이 포함된 전처리 함수 ---
def load_and_preprocess_siamese(path, img_size=(128, 128), augment=False):
    """이미지를 로드하고 전처리합니다. 학습 시 augmentation 적용."""
    try:
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size[0], img_size[1]])
        
        # 학습 시 Data Augmentation 적용
        if augment:
            # 랜덤 좌우 반전
            img = tf.image.random_flip_left_right(img)
            # 랜덤 밝기 조정 (±10%)
            img = tf.image.random_brightness(img, max_delta=0.1)
            # 랜덤 대비 조정
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            # 랜덤 색조 조정
            img = tf.image.random_hue(img, max_delta=0.05)
            # 랜덤 채도 조정
            img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
            # 값 범위 클리핑
            img = tf.clip_by_value(img, 0.0, 255.0)
        
        img = preprocess_input(img)
        return img
    except Exception as e:
        return tf.zeros([img_size[0], img_size[1], 3], dtype=tf.float32)



def create_base_network(input_shape, embedding_dim):
    """Embedding을 생성하는 Base Network를 구성."""
    base = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', 
        input_shape=input_shape
    )
    
    # [개선] Fine-tuning 레이어 수 감소 (50 → 25)
    # 과적합 방지를 위해 더 적은 레이어만 학습
    num_layers_to_fine_tune = 25
    
    for layer in base.layers[:-num_layers_to_fine_tune]:
        layer.trainable = False
    for layer in base.layers[-num_layers_to_fine_tune:]:
        layer.trainable = True

    inputs = base.input
    x = base.output
    
    # [개선] Dropout과 BatchNormalization, L2 Regularization 추가
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)  # 30% Dropout
    
    x = Dense(
        256, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.001),  # L2 정규화
        name='dense_256'
    )(x)
    
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)  # 20% Dropout
    
    x = Dense(
        embedding_dim, 
        activation=None,
        kernel_regularizer=regularizers.l2(0.001),  # L2 정규화
        name='embedding'
    )(x)
    
    outputs = Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    
    return Model(inputs, outputs, name='base_network')


# --- Triplet Loss 모델 ---
class TripletLossModel(Model):
    """Triplet Loss를 사용하는 Siamese Network 모델"""
    
    def __init__(self, base_network, margin, **kwargs):
        super().__init__(**kwargs)
        self.base_network = base_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="triplet_loss")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_triplet_loss")

    def call(self, inputs, training=None):
        return (
            self.base_network(inputs["anchor"], training=training),
            self.base_network(inputs["positive"], training=training),
            self.base_network(inputs["negative"], training=training),
        )

    def train_step(self, data):
        with tf.GradientTape() as tape:
            anchor_emb, positive_emb, negative_emb = self(data[0], training=True)
            
            pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
            loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, self.base_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_network.trainable_variables))
        self.loss_tracker.update_state(loss)
        
        return {"triplet_loss": self.loss_tracker.result()}

    def test_step(self, data):
        """검증 단계"""
        anchor_emb, positive_emb, negative_emb = self(data[0], training=False)
        
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        loss = tf.reduce_mean(loss)
        
        self.val_loss_tracker.update_state(loss)
        return {"triplet_loss": self.val_loss_tracker.result()}

    def on_epoch_begin(self, epoch, logs=None):
        """Epoch 시작 시 metric 초기화"""
        self.loss_tracker.reset_states()
        self.val_loss_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]

def get_triplet_generator(paths_by_class_map, class_list, img_size=(128, 128), 
                         augment=False, max_attempts=100):
    """
    Triplet (Anchor, Positive, Negative)를 생성하는 제너레이터
    augment=True일 때 데이터 증강 적용
    """
    while True:
        attempts = 0
        
        while attempts < max_attempts:
            anchor_class = random.choice(class_list)
            if paths_by_class_map[anchor_class] and len(paths_by_class_map[anchor_class]) >= 2:
                break
            attempts += 1
        
        if attempts >= max_attempts:
            print("Warning: Could not find valid anchor class")
            continue
            
        anchor_path = random.choice(paths_by_class_map[anchor_class])
        positive_path = random.choice(paths_by_class_map[anchor_class])
        
        while positive_path == anchor_path and len(paths_by_class_map[anchor_class]) > 1:
            positive_path = random.choice(paths_by_class_map[anchor_class])
        
        attempts = 0
        while attempts < max_attempts:
            negative_class = random.choice(class_list)
            if negative_class != anchor_class and paths_by_class_map[negative_class]:
                break
            attempts += 1
        
        if attempts >= max_attempts:
            print("Warning: Could not find valid negative class")
            continue
            
        negative_path = random.choice(paths_by_class_map[negative_class])
        
        yield (
            load_and_preprocess_siamese(anchor_path, img_size, augment=augment),
            load_and_preprocess_siamese(positive_path, img_size, augment=augment),
            load_and_preprocess_siamese(negative_path, img_size, augment=augment)
        )


def map_triplets_to_inputs(anchor, positive, negative):
    """Triplet을 모델 입력 형식으로 변환"""
    return ({"anchor": anchor, "positive": positive, "negative": negative}, 0.0)


# --- 메인 학습 함수 ---
def main():
    print("\n" + "="*60)
    print("Siamese Network (과적합 방지 버전) 학습 시작")
    print("="*60)
    
    # Mixed Precision 설정
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"Mixed Precision 활성화: {policy.name}")
    
    # GPU 메모리 최적화
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 설정 완료: {len(gpus)}개")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    
    # 설정 및 디렉토리
    cfg.create_directories()
    
    IMG_SIZE_SIAMESE = cfg.IMG_SIZE_SIAMESE
    print(f"Siamese Network 이미지 크기: {IMG_SIZE_SIAMESE}")
    
    train_dir = cfg.BASE_IMAGE_DIR / "train"
    test_dir = cfg.BASE_IMAGE_DIR / "test"
    
    # 데이터 스캔
    extensions = cfg.IMAGE_EXTENSIONS
    train_paths_by_class = {}
    test_paths_by_class = {}
    class_list = list(cfg.CLASSES)
    
    total_train = 0
    total_test = 0
    
    print("\n클래스별 이미지 파일 스캔 중...")
    for c in class_list:
        train_paths_by_class[c] = []
        for ext in extensions:
            train_paths_by_class[c].extend((train_dir / c).glob(ext))
        train_paths_by_class[c] = [str(p) for p in train_paths_by_class[c]]
        train_count = len(train_paths_by_class[c])
        total_train += train_count
        
        test_paths_by_class[c] = []
        for ext in extensions:
            test_paths_by_class[c].extend((test_dir / c).glob(ext))
        test_paths_by_class[c] = [str(p) for p in test_paths_by_class[c]]
        test_count = len(test_paths_by_class[c])
        total_test += test_count
        
        print(f"  {c:15} → Train: {train_count:4}, Test: {test_count:4}")
    
    print(f"\n총 이미지: Train={total_train}, Test={total_test}")
    
    if total_train == 0 or total_test == 0:
        print("\n오류: 학습 또는 테스트 이미지가 없습니다.")
        return
    
    # [개선] 데이터셋 생성 - 전체 데이터 사용
    BATCH_SIZE = 32  # 64 → 32로 줄임 (안정성 향상)
    
    output_sig = (
        tf.TensorSpec(shape=(*IMG_SIZE_SIAMESE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*IMG_SIZE_SIAMESE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*IMG_SIZE_SIAMESE, 3), dtype=tf.float32)
    )
    
    # Train 데이터셋 (augmentation 적용)
    dataset_siamese = tf.data.Dataset.from_generator(
        lambda: get_triplet_generator(train_paths_by_class, class_list, 
                                     IMG_SIZE_SIAMESE, augment=True),
        output_signature=output_sig
    )
    dataset_siamese = dataset_siamese.map(
        map_triplets_to_inputs, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    dataset_siamese = dataset_siamese.batch(BATCH_SIZE, drop_remainder=True)
    dataset_siamese = dataset_siamese.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Validation 데이터셋 (augmentation 없음)
    validation_dataset_siamese = tf.data.Dataset.from_generator(
        lambda: get_triplet_generator(test_paths_by_class, class_list, 
                                     IMG_SIZE_SIAMESE, augment=False),
        output_signature=output_sig
    )
    validation_dataset_siamese = validation_dataset_siamese.map(
        map_triplets_to_inputs,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    validation_dataset_siamese = validation_dataset_siamese.batch(BATCH_SIZE, drop_remainder=True)
    validation_dataset_siamese = validation_dataset_siamese.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print("데이터셋 준비 완료 (Data Augmentation 적용)")
    
    # 모델 생성
    EMBEDDING_DIM = 128
    MARGIN = 0.5
    
    print(f"\n모델 생성 중... (Embedding={EMBEDDING_DIM}, Margin={MARGIN})")
    base_network = create_base_network((*IMG_SIZE_SIAMESE, 3), EMBEDDING_DIM)
    siamese_model = TripletLossModel(base_network, MARGIN)
    
    # [개선] 학습률 감소 (1e-4 → 5e-5)
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        jit_compile=True
    )
    
    print("\nBase Network 구조:")
    trainable_count = sum([1 for layer in base_network.layers if layer.trainable])
    print(f"  총 레이어: {len(base_network.layers)}")
    print(f"  학습 가능: {trainable_count}")
    print(f"  동결: {len(base_network.layers) - trainable_count}")
    
    # [개선] 콜백 설정
    base_network_save_path = cfg.MODEL_SAVE_DIR / "siamesenetwork" / "base_network_best.h5"
    
    callbacks_list = [
        EarlyStopping(
            monitor='val_triplet_loss',
            mode='min',
            verbose=1,
            patience=10,  # 5 → 10으로 늘림
            restore_best_weights=False
        ),
        # Learning Rate 감소 콜백 추가
        ReduceLROnPlateau(
            monitor='val_triplet_loss',
            factor=0.5,  # 학습률을 절반으로
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        SaveBaseNetworkCallback(
            base_network_save_path,
            monitor='val_triplet_loss',
            mode='min'
        )
    ]
    
    # [개선] 학습 - 전체 데이터 사용
    EPOCHS = 50
    # 전체 데이터를 사용하도록 steps 계산
    STEPS = (total_train + BATCH_SIZE - 1) // BATCH_SIZE
    VALIDATION_STEPS = (total_test + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n학습 시작:")
    print(f"  이미지 크기: {IMG_SIZE_SIAMESE}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps/Epoch: {STEPS} (전체 데이터)")
    print(f"  Validation Steps: {VALIDATION_STEPS} (전체 데이터)")
    print(f"  Fine-tuning 레이어: 25개 (과적합 방지)")
    print(f"  Dropout: 0.3, 0.2")
    print(f"  L2 Regularization: 0.001")
    print(f"  Data Augmentation: 적용")
    print("="*60 + "\n")
    
    history = siamese_model.fit(
        dataset_siamese,
        epochs=EPOCHS,
        steps_per_epoch=STEPS,
        validation_data=validation_dataset_siamese,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 결과 저장
    print("\n" + "="*60)
    print("[Siamese Network] 학습 완료")
    print("="*60)
    print(f"최적의 Base Network 모델 저장 완료: {base_network_save_path}")
    
    plot_save_path = cfg.MODEL_SAVE_DIR / "siamesenetwork" / "siamese_loss_plot.png"
    plot_siamese_history(history, plot_save_path)
    
    print("\n모든 작업 완료!")


if __name__ == "__main__":
    from common_utility import measure_process_time
    measure_process_time(main)