import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import mixed_precision, regularizers
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random

import common_config as cfg


# --- GPU 최적화 설정 ---
def setup_gpu_optimization():
    """GPU 최적화 설정"""
     
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print("XLA JIT 컴파일 활성화")
            
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
    else:
        print("GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다.")
    
    return len(gpus) > 0


# --- 시각화 함수 ---
def plot_mobilenet_history(history, save_path):
    
    plt.figure(figsize=(7, 5))
    
    plt.plot(history.history['triplet_loss'], label='Train Loss')
    if 'val_triplet_loss' in history.history:
        plt.plot(history.history['val_triplet_loss'], label='Validation Loss')
    
    plt.title('MobileNetV3 (Triplet) Model Loss')
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
    """Epoch 종료 시 최적 모델 저장"""
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
            print(f"\nWarning: {self.monitor} not found in logs")
            return

        improved = (self.mode == 'min' and current < self.best) or \
                   (self.mode == 'max' and current > self.best)
        
        if improved:
            print(f"\nEpoch {epoch+1}: {self.monitor} improved {self.best:.5f} → {current:.5f}")
            self.best = current
            try:
                self.model.base_network.save(self.save_path)
                print(f"Model saved: {self.save_path}")
            except Exception as e:
                print(f"Save error: {e}")


# --- 최적화된 데이터 전처리 함수 ---
@tf.function
def load_and_preprocess_mobilenet(path, augment=False):
    """
    GPU에서 실행되는 최적화된 이미지 전처리
    @tf.function 데코레이터로 그래프 모드 최적화
    """
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])  # shape 명시
    img = tf.image.resize(img, [224, 224])
    
    if augment:
        # 모든 augmentation을 하나의 그래프로 결합
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.image.random_hue(img, max_delta=0.05)
        img = tf.image.random_saturation(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, 0.0, 255.0)
    
    img = preprocess_input(img)
    return img


def create_triplet_dataset(paths_by_class, class_list, img_size=(224, 224), 
                          augment=False, batch_size=256, is_training=True):
    """
    최적화된 Triplet 데이터셋 생성
    Generator 대신 tf.data API 직접 활용
    """
    def triplet_generator():
        while True:
            # Anchor 클래스 선택
            anchor_class = random.choice(class_list)
            if not paths_by_class[anchor_class] or len(paths_by_class[anchor_class]) < 2:
                continue
            
            # Anchor, Positive 선택
            anchor_path = random.choice(paths_by_class[anchor_class])
            positive_path = random.choice(paths_by_class[anchor_class])
            
            while positive_path == anchor_path and len(paths_by_class[anchor_class]) > 1:
                positive_path = random.choice(paths_by_class[anchor_class])
            
            # Negative 선택
            negative_class = random.choice([c for c in class_list if c != anchor_class])
            if not paths_by_class[negative_class]:
                continue
            
            negative_path = random.choice(paths_by_class[negative_class])
            
            yield (anchor_path, positive_path, negative_path)
    
    # 데이터셋 생성
    output_sig = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string)
    )
    
    dataset = tf.data.Dataset.from_generator(
        triplet_generator,
        output_signature=output_sig
    )
    
    # 병렬 전처리 (핵심 최적화!)
    def process_triplet(anchor_path, positive_path, negative_path):
        anchor = load_and_preprocess_mobilenet(anchor_path, augment=augment)
        positive = load_and_preprocess_mobilenet(positive_path, augment=augment)
        negative = load_and_preprocess_mobilenet(negative_path, augment=augment)
        return {"anchor": anchor, "positive": positive, "negative": negative}, 0.0
    
    # 최적화된 파이프라인 구성
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    
    # 병렬 처리 (CPU 코어 수만큼)
    dataset = dataset.map(
        process_triplet,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # 배치 처리
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch로 GPU 대기시간 제거
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


def create_base_network_mobilenet(input_shape, embedding_dim):
    """Embedding을 생성하는 Base Network (MobileNetV3)"""
    base = MobileNetV3Large(
        weights='imagenet', 
        include_top=False, 
        pooling='avg', 
        input_shape=input_shape
    )
    
    # Fine-tuning 레이어 수
    num_layers_to_fine_tune = 25
    
    for layer in base.layers[:-num_layers_to_fine_tune]:
        layer.trainable = False
    for layer in base.layers[-num_layers_to_fine_tune:]:
        layer.trainable = True

    inputs = base.input
    x = base.output
    
    # Mixed Precision 대응
    x = BatchNormalization(name='bn_1', dtype='float32')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    
    x = Dense(
        256, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_256',
        dtype='float32'  # Mixed Precision 명시
    )(x)
    
    x = BatchNormalization(name='bn_2', dtype='float32')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    
    x = Dense(
        embedding_dim, 
        activation=None,
        kernel_regularizer=regularizers.l2(0.001),
        name='embedding',
        dtype='float32'  # Mixed Precision 명시
    )(x)
    
    outputs = Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name='l2_norm')(x)
    
    return Model(inputs, outputs, name='base_network')


# --- Triplet Loss 모델 (최적화) ---
class TripletLossModel(Model):
    """최적화된 Triplet Loss 모델"""
    
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

    def compute_loss(self, anchor_emb, positive_emb, negative_emb):
        """Loss 계산 최적화"""
        pos_dist = tf.reduce_sum(tf.square(anchor_emb - positive_emb), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor_emb - negative_emb), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            anchor_emb, positive_emb, negative_emb = self(data[0], training=True)
            loss = self.compute_loss(anchor_emb, positive_emb, negative_emb)
        
        gradients = tape.gradient(loss, self.base_network.trainable_variables)
        
        # Gradient clipping으로 학습 안정화
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g 
                    for g in gradients]
        
        self.optimizer.apply_gradients(
            zip(gradients, self.base_network.trainable_variables)
        )
        self.loss_tracker.update_state(loss)
        
        return {"triplet_loss": self.loss_tracker.result()}

    def test_step(self, data):
        anchor_emb, positive_emb, negative_emb = self(data[0], training=False)
        loss = self.compute_loss(anchor_emb, positive_emb, negative_emb)
        self.val_loss_tracker.update_state(loss)
        return {"triplet_loss": self.val_loss_tracker.result()}

    def on_epoch_begin(self, epoch, logs=None):
        self.loss_tracker.reset_states()
        self.val_loss_tracker.reset_states()

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker]


# --- 메인 학습 함수 ---
def main():
    print("\n" + "="*60)
    print("MobileNetV3 (Triplet Loss) 최적화 학습 시작")
    print("="*60)
    
    # GPU 최적화 설정
    has_gpu = setup_gpu_optimization()
    
    # 설정 및 디렉토리
    cfg.create_directories()
    IMG_SIZE_MOBILENET = cfg.IMG_SIZE_MOBILENET 
    print(f"이미지 크기: {IMG_SIZE_MOBILENET}")
    
    train_dir = cfg.BASE_IMAGE_DIR / "train"
    validation_dir = cfg.BASE_IMAGE_DIR / "validation"
    
    # 데이터 스캔
    extensions = cfg.IMAGE_EXTENSIONS
    train_paths_by_class = {}
    validation_paths_by_class = {}
    class_list = list(cfg.CLASSES)
    
    total_train = 0
    total_validation = 0
    
    print("\n클래스별 이미지 파일 스캔 중...")
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
        
        print(f"  {c:15} → Train: {train_count:4}, Validation: {validation_count:4}")
    
    print(f"\n총 이미지: Train={total_train}, Validation={total_validation}")
    
    if total_train == 0 or total_validation == 0:
        print("\n오류: 학습 또는 테스트 이미지가 없습니다.")
        return
    
    # 배치 크기 최적화 (GPU 메모리에 따라 조정)
    BATCH_SIZE = 256 if has_gpu else 32
    print(f"배치 크기: {BATCH_SIZE}")
    
    # 최적화된 데이터셋 생성
    print("\n데이터셋 구성 중...")
    train_dataset = create_triplet_dataset(
        train_paths_by_class, 
        class_list, 
        IMG_SIZE_MOBILENET,
        augment=True, 
        batch_size=BATCH_SIZE,
        is_training=True
    )
    
    val_dataset = create_triplet_dataset(
        validation_paths_by_class, 
        class_list, 
        IMG_SIZE_MOBILENET,
        augment=False, 
        batch_size=BATCH_SIZE,
        is_training=False
    )
    
    print("데이터셋 준비 완료 (최적화된 파이프라인)")
    
    # 모델 생성
    EMBEDDING_DIM = 128
    MARGIN = 0.5
    
    print(f"\n모델 생성 중... (Embedding={EMBEDDING_DIM}, Margin={MARGIN})")
    base_network = create_base_network_mobilenet((*IMG_SIZE_MOBILENET, 3), EMBEDDING_DIM)
    mobilenet_model = TripletLossModel(base_network, MARGIN)
    
    # Optimizer 최적화
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-5,
        clipnorm=1.0   
    )
    
    mobilenet_model.compile(
        optimizer=optimizer,
        jit_compile=True  
    )
    
    print("\nBase Network 구조:")
    trainable_count = sum([1 for layer in base_network.layers if layer.trainable])
    print(f"  총 레이어: {len(base_network.layers)}")
    print(f"  학습 가능: {trainable_count}")
    print(f"  동결: {len(base_network.layers) - trainable_count}")
    
    # 콜백 설정
    base_network_save_path = cfg.MODEL_SAVE_DIR / "mobilenetv3" / cfg.SEED_DIR / "base_network_best.h5"
    base_network_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks_list = [
        EarlyStopping(
            monitor='val_triplet_loss',
            mode='min',
            verbose=1,
            patience=10,
            restore_best_weights=False
        ),
        ReduceLROnPlateau(
            monitor='val_triplet_loss',
            factor=0.5,
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
    
    # 학습
    EPOCHS = 50
    STEPS = (total_train + BATCH_SIZE - 1) // BATCH_SIZE
    VALIDATION_STEPS = (total_validation + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n학습 시작:")
    print(f"  이미지 크기: {IMG_SIZE_MOBILENET}")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps/Epoch: {STEPS}")
    print(f"  Validation Steps: {VALIDATION_STEPS}")
    print(f"  Fine-tuning 레이어: 25개")
    print(f"  XLA JIT 컴파일: 활성화")
    print(f"  Mixed Precision: 활성화")
    print("="*60 + "\n")
    
    history = mobilenet_model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS,
        validation_data=val_dataset,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 결과 저장
    print("\n" + "="*60)
    print(" [MobileNetV3] 학습 완료")
    print("="*60)
    print(f" 최적의 Base Network 모델 저장 완료: {base_network_save_path}")
    
    plot_save_path = cfg.MODEL_SAVE_DIR / "mobilenetv3" / cfg.SEED_DIR / "mobilenet_loss_plot.png"
    plot_mobilenet_history(history, plot_save_path)
    
    print("\n 모든 작업 완료!")


if __name__ == "__main__":
 
   cfg.measure_process_time(main)