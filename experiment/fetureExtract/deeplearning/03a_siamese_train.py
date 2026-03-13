### ===================================================
### 03a. Siamese Network - GPU 사용률 최적화 버전
### ===================================================

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Dropout, BatchNormalization
 
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess

from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import mixed_precision, regularizers
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random

import common_config as cfg

from common_utility import FLOPSCalculator


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
            return

        improved = (self.mode == 'min' and current < self.best) or \
                   (self.mode == 'max' and current > self.best)
        
        if improved:
            print(f"\n✓ Epoch {epoch+1}: {self.monitor} {self.best:.5f} → {current:.5f}")
            self.best = current
            try:
                self.model.base_network.save(self.save_path)
            except Exception as e:
                print(f"  ✗ Save error: {e}")


# --- [핵심] TensorFlow 네이티브 데이터 로딩 ---
@tf.function
def load_and_preprocess_image_tf(path, img_size, augment=False):
    """
    TensorFlow 네이티브 함수로 이미지 로딩 (GPU에서 실행 가능)
    Python 없이 순수 TensorFlow 연산만 사용
    """
    # 이미지 읽기
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)  
    img = tf.image.resize(img, img_size)
    
    # Data Augmentation (조건부)
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 255.0)
    
    # Preprocessing
    img = mobilenet_preprocess(img)
    return img


def create_base_network(input_shape, embedding_dim):
    """Embedding을 생성하는 Base Network를 구성."""
    base = MobileNetV3Small(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', 
        input_shape=input_shape
    )
    
    num_layers_to_fine_tune = 25
    
    for layer in base.layers[:-num_layers_to_fine_tune]:
        layer.trainable = False
    for layer in base.layers[-num_layers_to_fine_tune:]:
        layer.trainable = True

    inputs = base.input
    x = base.output
    
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    
    x = Dense(
        256, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_256'
    )(x)
    
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    
    x = Dense(
        embedding_dim, 
        activation=None,
        kernel_regularizer=regularizers.l2(0.001),
        name='embedding'
    )(x)
    
    outputs = Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    
    return Model(inputs, outputs, name='base_network')


# --- Triplet Loss 모델 ---
class TripletLossModel(Model):
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
        anchor_emb, positive_emb, negative_emb = self(data[0], training=False)
        
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        loss = tf.reduce_mean(loss)
        
        self.val_loss_tracker.update_state(loss)
        return {"triplet_loss": self.val_loss_tracker.result()}

    def on_epoch_begin(self, epoch, logs=None):
        self.loss_tracker.reset_states()
        self.val_loss_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]


# --- [핵심] 효율적인 Triplet 데이터셋 생성 ---
def create_triplet_dataset_optimized(paths_by_class, class_list, img_size, 
                                    batch_size, augment=False, prefetch_batches=10):
    """
    GPU 병목을 최소화하는 최적화된 Triplet 데이터셋
    핵심: 미리 여러 배치를 준비해두고 GPU가 쉬지 않게 함
    """
    
    # 1. 모든 경로를 텐서로 변환 (메모리에 로드)
    all_paths_by_class = {}
    for cls in class_list:
        all_paths_by_class[cls] = tf.constant(paths_by_class[cls], dtype=tf.string)
    
    class_indices = {cls: i for i, cls in enumerate(class_list)}
    num_classes = len(class_list)
    
    def triplet_generator():
        """빠른 triplet 생성 (Python으로 한 번만 실행)"""
        while True:
            # Anchor 클래스 선택
            anchor_class = random.choice(class_list)
            anchor_paths = paths_by_class[anchor_class]
            
            if len(anchor_paths) < 2:
                continue
            
            # Anchor와 Positive 선택
            anchor_idx, pos_idx = random.sample(range(len(anchor_paths)), 2)
            
            # Negative 클래스 선택
            negative_classes = [c for c in class_list if c != anchor_class]
            negative_class = random.choice(negative_classes)
            negative_paths = paths_by_class[negative_class]
            
            if len(negative_paths) == 0:
                continue
            
            neg_idx = random.randint(0, len(negative_paths) - 1)
            
            yield {
                'anchor': anchor_paths[anchor_idx],
                'positive': anchor_paths[pos_idx],
                'negative': negative_paths[neg_idx]
            }
    
    # 2. Dataset 생성
    dataset = tf.data.Dataset.from_generator(
        triplet_generator,
        output_signature={
            'anchor': tf.TensorSpec(shape=(), dtype=tf.string),
            'positive': tf.TensorSpec(shape=(), dtype=tf.string),
            'negative': tf.TensorSpec(shape=(), dtype=tf.string)
        }
    )
    
    # 3. [핵심] 이미지 로딩을 병렬 처리 (CPU 멀티스레딩)
    def load_triplet(triplet_paths):
        anchor = load_and_preprocess_image_tf(triplet_paths['anchor'], img_size, augment)
        positive = load_and_preprocess_image_tf(triplet_paths['positive'], img_size, augment)
        negative = load_and_preprocess_image_tf(triplet_paths['negative'], img_size, augment)
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }, tf.constant(0.0)
    
    # 4. [최적화] 병렬 로딩 + 배치 + Prefetch
    dataset = dataset.map(
        load_triplet,
        num_parallel_calls=tf.data.AUTOTUNE,  # CPU 코어 수만큼 병렬 처리
        deterministic=False  # 순서 무시하고 빠른 것부터
    )
    
    # 5. [최적화] 배치 전에 캐싱 (메모리 여유 있으면)
    # dataset = dataset.cache()  # 메모리 부족 시 주석 처리
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # 6. Prefetch - GPU가 처리하는 동안 다음 배치 준비
    dataset = dataset.prefetch(buffer_size=prefetch_batches)
    
    return dataset


# --- 메인 학습 함수 ---
def main():
    print("\n" + "="*60)
    print("Siamese Network (GPU 최적화) 학습 시작")
    print("="*60)
    
    # Mixed Precision 설정
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f" Mixed Precision: {policy.name}")
    
    # GPU 메모리 최적화
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU: {len(gpus)}개 감지")
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    
    # 설정
    cfg.create_directories()
    IMG_SIZE_SIAMESE = cfg.IMG_SIZE_SIAMESE
    
    train_dir = cfg.BASE_IMAGE_DIR / "train"
    validation_dir = cfg.BASE_IMAGE_DIR / "validation"
    
    # 데이터 스캔
    extensions = cfg.IMAGE_EXTENSIONS
    train_paths_by_class = {}
    validation_paths_by_class = {}
    class_list = list(cfg.CLASSES)
    
    total_train = 0
    total_validation = 0
    
    print("\n이미지 파일 스캔 중...")
    for c in class_list:
        train_paths_by_class[c] = []
        for ext in extensions:
            train_paths_by_class[c].extend((train_dir / c).glob(ext))
        train_paths_by_class[c] = [str(p) for p in train_paths_by_class[c]]
        train_count = len(train_paths_by_class[c])
        total_train += train_count
        
        validation_paths_by_class[c] = []
        for ext in extensions:
            validation_paths_by_class[c].extend((validation_dir / c).glob(ext))
        validation_paths_by_class[c] = [str(p) for p in validation_paths_by_class[c]]
        validation_count = len(validation_paths_by_class[c])
        total_validation += validation_count
        
        print(f"  {c:15} → Train: {train_count:4}, Val: {validation_count:4}")
    
    print(f"\n총 이미지: Train={total_train}, Val={total_validation}")
    
    if total_train == 0 or total_validation == 0:
        print("\n오류: 이미지가 없습니다.")
        return
    
    # [최적화] 데이터셋 생성
    BATCH_SIZE = 64  # GPU 사용률 높이기 위해 배치 크기 증가
    PREFETCH_BATCHES = 20  # GPU가 처리하는 동안 20개 배치 미리 준비
    
    print(f"\n최적화된 데이터셋 생성 중...")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Prefetch Batches: {PREFETCH_BATCHES}")
    
    # Train 데이터셋
    dataset_siamese = create_triplet_dataset_optimized(
        train_paths_by_class,
        class_list,
        IMG_SIZE_SIAMESE,
        BATCH_SIZE,
        augment=True,
        prefetch_batches=PREFETCH_BATCHES
    )
    
    # Validation 데이터셋
    validation_dataset_siamese = create_triplet_dataset_optimized(
        validation_paths_by_class,
        class_list,
        IMG_SIZE_SIAMESE,
        BATCH_SIZE,
        augment=False,
        prefetch_batches=10
    )
    
    print("데이터셋 준비 완료")
    
    # 모델 생성
    EMBEDDING_DIM = 128
    MARGIN = 0.5
    
    print(f"\n모델 생성 중...")
    base_network = create_base_network((*IMG_SIZE_SIAMESE, 3), EMBEDDING_DIM)
    siamese_model = TripletLossModel(base_network, MARGIN)
    
    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        jit_compile=True  # XLA 컴파일
    )
    
    print("\nBase Network:")
    trainable = sum([1 for l in base_network.layers if l.trainable])
    print(f"   총 레이어: {len(base_network.layers)}")
    print(f"   학습 가능: {trainable}")
    
    # 콜백
    save_path = cfg.MODEL_SAVE_DIR / "siamesenetwork" / cfg.SEED_DIR / "base_network_best.h5"
    
    callbacks_list = [
        EarlyStopping(
            monitor='val_triplet_loss',
            patience=10,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_triplet_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        SaveBaseNetworkCallback(save_path)
    ]
    
    # 학습
    EPOCHS = 50
    STEPS = (total_train + BATCH_SIZE - 1) // BATCH_SIZE
    VAL_STEPS = (total_validation + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n학습 시작:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Steps/Epoch: {STEPS}")
    print(f"   Validation Steps: {VAL_STEPS}")
    print(f"   최적화: 병렬 로딩 + Prefetch + Mixed Precision")
    print("="*60 + "\n")
    
    history = siamese_model.fit(
        dataset_siamese,
        epochs=EPOCHS,
        steps_per_epoch=STEPS,
        validation_data=validation_dataset_siamese,
        validation_steps=VAL_STEPS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 결과 저장
    print("\n" + "="*60)
    print("학습 완료")
    print("="*60)
    print(f"모델 저장: {save_path}")
    
    # 7-1. FLOPS 측정
    # Siamese triplet dataset은 {"anchor","positive","negative"} 딕셔너리 구조라
    # base_network(단일 이미지 입력)에 직접 넣을 수 없으므로,
    # FLOPS 측정용 단일 이미지 dataset을 별도로 생성
    all_train_paths = []
    for c in class_list:
        all_train_paths.extend(train_paths_by_class[c])

    flops_dataset = tf.data.Dataset.from_tensor_slices(all_train_paths)
    flops_dataset = flops_dataset.map(
        lambda path: (load_and_preprocess_image_tf(path, IMG_SIZE_SIAMESE, augment=False), 0.0),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    flops_dataset = flops_dataset.batch(BATCH_SIZE, drop_remainder=True)
    flops_dataset = flops_dataset.prefetch(tf.data.AUTOTUNE)

    flops_calculator = FLOPSCalculator()
    flops = flops_calculator.calculate_with_dataset(
        base_network,
        input_shape=(1, *cfg.IMG_SIZE_SIAMESE, 3),
        model_name="Siamese Network",
        dataset=flops_dataset,
        num_samples=total_train,
        num_batches=total_train // BATCH_SIZE
    )
    print(flops)
    print(f"{'='*60}\n")
    
    plot_path = cfg.MODEL_SAVE_DIR / "siamesenetwork" / cfg.SEED_DIR / "siamese_loss_plot.png"
    plot_siamese_history(history, plot_path)
    
    print("\n모든 작업 완료!")


if __name__ == "__main__":

    cfg.measure_process_time(main)