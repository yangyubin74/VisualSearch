import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from thop import profile, clever_format
 

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import config 
# ====================================================================
# 1. 실제 이미지 로드 및 전처리 함수
# ====================================================================
def load_image_as_tensor(image_path, target_size=(224, 224), is_flatten=False):
    """
    실제 이미지 파일을 읽어서 모델 입력용 텐서로 변환
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple): 모델이 요구하는 입력 크기 (예: 224, 224)
        is_flatten (bool): Autoencoder 처럼 1차원 입력이 필요한 경우 True
    """
    if not os.path.exists(image_path):
        # 파일이 없으면 랜덤 텐서 반환 (테스트용)
        print(f"경고: '{image_path}' 파일을 찾을 수 없습니다. 임의의 데이터를 생성합니다.")
        c, h, w = 3, target_size[0], target_size[1]
        if is_flatten:
            return torch.randn(1, c * h * w) # 예: 784
        return torch.randn(1, c, h, w)

    # 1. 이미지 열기
    input_image = Image.open(image_path).convert('RGB')

    # 2. 전처리 파이프라인 정의 (일반적인 CNN 모델용)
    preprocess = transforms.Compose([
        transforms.Resize(target_size),      # 모델 입력 크기로 조절
        transforms.ToTensor(),               # 텐서로 변환 (0~1 정규화 포함)
        # ImageNet 학습 모델 표준 정규화 (선택사항이나 실제 추론시 필수)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    
    # 3. Autoencoder용 Flatten 처리 (예: 28x28 -> 784)
    if is_flatten:
        # Autoencoder 예제가 흑백(1ch) 28x28=784 입력이라고 가정할 때
        gray_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        input_tensor = gray_transform(input_image)
        input_tensor = input_tensor.view(-1) # [1, 784] 형태로 평탄화
        return input_tensor.unsqueeze(0)     # Batch 차원 추가 -> [1, 784]

    # 4. 배치 차원 추가 (C, H, W) -> (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0) 
    return input_batch

def load_image_as_tensor_siames(image_path, target_size=(105, 105)):
    
    if not os.path.exists(image_path):
        print(f"경고: '{image_path}' 없음. 랜덤 텐서 생성.")
        return torch.randn(1, 3, target_size[0], target_size[1])

    input_image = Image.open(image_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0) # (1, C, H, W)
# ====================================================================
# 2. FLOPS 측정 함수 (학습 고려)
# ====================================================================
def measure_training_flops(model, input_tensor, model_name="Model"):
    model.eval() 
    
    # Forward FLOPS (thop)
    flops_forward, params = profile(model, inputs=(input_tensor,), verbose=False)
    
    # Training FLOPS 근사치 (Forward * 3)
    flops_train = flops_forward * 3
    
    # 포맷팅
    flops_f_str, params_str = clever_format([flops_forward, params], "%.3f")
    flops_t_str = clever_format([flops_train], "%.3f")
    
    print(f"[{model_name}]")
    print(f" - Input Shape     : {list(input_tensor.shape)}")
    print(f" - Inference FLOPs : {flops_f_str}")
    print(f" - Training FLOPs  : {flops_t_str} (Estimated x3)")
    print(f" - Parameters      : {params_str}")
    print("-" * 50)

# ====================================================================
# 3. 모델 정의 (Autoencoder)
# ====================================================================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16 (Latent)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 128x128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # -> 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # -> 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # -> 16x16
        )
        
        # Decoder: 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid() # 0~1 사이 값 복원
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ====================================================================
# 2. 이미지 로드 함수 (128x128 설정)
# ====================================================================
def load_image_128(image_path):
    if not os.path.exists(image_path):
        # 파일 없으면 랜덤 생성 (3채널, 128, 128)
        return torch.randn(1, 3, 128, 128)

    input_image = Image.open(image_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),  # ★ 128x128로 리사이징
        transforms.ToTensor(),
        # Autoencoder는 보통 입력 그대로 복원하므로 Normalize는 상황에 따라 뺍니다.
        # 여기서는 0~1 범위를 유지하기 위해 ToTensor만 사용하거나,
        # 성능을 위해 Normalize를 했다면 Decoder 끝에 역변환이 필요할 수 있습니다.
    ])

    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0) # (1, 3, 128, 128)

# ====================================================================
# 3. FLOPS 측정 함수
# ====================================================================
def measure_flops_128(image_path):
    print("="*60)
    print("Convolutional Autoencoder (128x128 Input)")
    print("="*60)

    # 모델 생성
    model = ConvAutoencoder()
    model.eval()

    # 이미지 로드 (128x128)
    input_tensor = load_image_128(image_path)
    
    print(f"Input Shape: {list(input_tensor.shape)}") # [1, 3, 128, 128] 확인

    # Forward FLOPS 측정
    flops_forward, params = profile(model, inputs=(input_tensor,), verbose=False)
    
    # 학습(Training) FLOPS 추정 (x3)
    flops_train = flops_forward * 3

    # 출력 포맷팅
    flops_f_str, params_str = clever_format([flops_forward, params], "%.3f")
    flops_t_str = clever_format([flops_train], "%.3f")

    print(f"Parameters      : {params_str}")
    print("-" * 50)
    print(f"Inference FLOPs : {flops_f_str}")
    print(f"Training FLOPs  : {flops_t_str} (Estimated x3)")
    print("-" * 50)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 공유되는 CNN 백본
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
        )
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 256), # 최종 임베딩 벡터
        )
    
    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def forward(self, input1, input2):
        # 샴 네트워크는 백본을 두 번 통과
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def measure_siamese_training_flops(image_path):
    print("="*60)
    print("Siamese Network (Pair Image Training)")
    print("="*60)

    # 1. 모델 준비
    model = SiameseNetwork()
    model.eval()

    # 2. 데이터 준비 (Siamese는 이미지 2장이 필요)
    # 실제로는 Positive/Negative Pair를 쓰지만, FLOPS 측정엔 같은 이미지를 2번 넣어도 무관
    img1 = load_image_as_tensor(image_path, target_size=(105, 105))
    img2 = load_image_as_tensor(image_path, target_size=(105, 105)) # 동일 크기 복제

    print(f"Input Shape (Pair): {list(img1.shape)} x 2")

    # 3. Forward FLOPS 측정
    # ★ 중요: inputs에 튜플 형태로 (img1, img2)를 넘겨야 forward(input1, input2)가 호출됨
    flops_forward, params = profile(model, inputs=(img1, img2), verbose=False)

    # 4. Training FLOPS 계산 (Forward * 3)
    flops_train = flops_forward * 3

    # 5. 결과 출력
    flops_f_str, params_str = clever_format([flops_forward, params], "%.3f")
    flops_t_str = clever_format([flops_train], "%.3f")

    print(f"Parameters      : {params_str}")
    print("-" * 50)
    print(f"Inference FLOPs : {flops_f_str} (이미지 2장 처리 합계)")
    print(f"Training FLOPs  : {flops_t_str} (Estimated x3)")
    print("-" * 50)
    print("※ 참고: Siamese Network는 한 번의 학습 Step(Batch)에서 \n"
            "   두 개의 이미지가 백본을 통과하므로 단일 모델보다 연산량이 약 2배 높게 나옵니다.")
# ====================================================================
# 4. 메인 실행
# ====================================================================
if __name__ == "__main__":
    # ★ 여기에 테스트할 실제 이미지 경로를 입력하세요 ★
    
    TEST_IMAGE_PATH =config.MODEL_BASE_IMAGE_DIR + "/train/dress/27cefd2a-7679-6a45-0500-6b07d6b56c4e.png"
    
    # 테스트용 더미 이미지 생성 (실제 파일이 없을 경우를 대비해)
    if not os.path.exists(TEST_IMAGE_PATH):
        dummy = Image.new('RGB', (512, 512), color='red')
        dummy.save(TEST_IMAGE_PATH)
        print(f"테스트용 이미지 생성됨: {TEST_IMAGE_PATH}")

    print("="*60)
    print(f"실제 이미지 파일 로드: {TEST_IMAGE_PATH}")
    print("="*60 + "\n")

 

    # 2. EfficientNet-B0 (224x224)
    model_eff = models.efficientnet_b0(pretrained=False)
    input_eff = load_image_as_tensor(TEST_IMAGE_PATH, target_size=config.IMG_SIZE_EFFICIENTNET)
    measure_training_flops(model_eff, input_eff, "EfficientNet-B0")

    # 3. MobileNetV3 (224x224)
    model_mobile = models.mobilenet_v3_large(pretrained=False)
    input_mobile = load_image_as_tensor(TEST_IMAGE_PATH, target_size=config.IMG_SIZE_MOBILENET)
    measure_training_flops(model_mobile, input_mobile, "MobileNetV3-Large")
    
    # 4. Siamese Network (105x105)
    measure_siamese_training_flops(TEST_IMAGE_PATH)
    
    # 5. Convolutional Autoencoder (128x128)
    measure_flops_128(TEST_IMAGE_PATH)