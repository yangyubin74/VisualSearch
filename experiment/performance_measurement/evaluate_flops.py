import torch
import torchvision.models as models
import torch.nn as nn
import sys
import subprocess

# 1. thop 라이브러리 설치 (터미널에서 먼저 실행 권장)
# pip install thop
try:
    from thop import profile
except ImportError:
    print("thop 라이브러리를 설치합니다...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop"])
    from thop import profile

# --- 🚨 중요: 사용자 정의 모델 ---
# 아래 클래스들은 예시입니다.
# 사용자의 실제 Autoencoder와 SiameseNetwork 모델 정의로 교체해야 합니다.

class YourAutoencoder(nn.Module):
    def __init__(self):
        super(YourAutoencoder, self).__init__()
        # 🚨 예시: 사용자의 실제 '인코더' 레이어로 교체하세요.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        # '특징 추출'을 비교하는 것이므로 인코더 부분만 계산합니다.
        return self.encoder(x)

class YourSiameseNetwork(nn.Module):
    def __init__(self):
        super(YourSiameseNetwork, self).__init__()
        # 🚨 예시: 사용자의 실제 '백본(backbone)' 네트워크로 교체하세요.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 128 * 128, 128) # 512x512 입력 기준
        )

    def forward_one(self, x):
        # 1개 이미지를 받아 특징 벡터를 반환하는 부분
        return self.backbone(x)

    def forward(self, input1, input2=None):
        # 특징 추출 시에는 입력 1개만 처리
        if input2 is None:
            return self.forward_one(input1)
        
        # (훈련 시)
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# -----------------------------------

# 2. 더미 입력 텐서 생성 (배치 1, 3채널, 512x512 크기)
# 🚨🚨🚨 사용자의 실제 실험(특징 추출) 시의 이미지 크기와 동일해야 합니다!
input_size = (1, 3, 512, 512)
dummy_input = torch.randn(input_size)
print(f"Using Dummy Input Size: {dummy_input.shape}\n")

# 3. 프로파일링할 모델 목록
models_to_profile = {
    # 🚨 EfficientNet-B0 대신 실제 사용한 버전(예: efficientnet_b5())으로 변경하세요.
    "EfficientNet-B0": models.efficientnet_b0(), 
    "MobileNetV3-Large": models.mobilenet_v3_large(),
    "YourAutoencoder (Encoder)": YourAutoencoder(),
    "YourSiameseNetwork (Backbone)": YourSiameseNetwork()
}

results = {}

# 4. 각 모델별 FLOPs 및 파라미터 계산
for model_name, model in models_to_profile.items():
    model.eval() # 평가 모드로 설정
    
    inputs_tuple = (dummy_input, )
    
    try:
        flops, params = profile(model, inputs=inputs_tuple, verbose=False)
        results[model_name] = {
            "FLOPs (G)": flops / 1e9,  # GFLOPs (10억 단위)
            "Params (M)": params / 1e6   # 파라미터 (백만 단위)
        }
    except Exception as e:
        results[model_name] = {"Error": str(e)}

# 5. 결과 출력
print("--- FLOPs 및 파라미터 수 ---")
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    if "Error" in metrics:
        print(f"  Error calculating FLOPs: {metrics['Error']}")
    else:
        # GFLOPs = Giga FLOPs (10억 번 연산)
        print(f"  FLOPs: {metrics['FLOPs (G)']:.2f} G")
        # M Params = Mega Params (백만 개 파라미터)
        print(f"  Params: {metrics['Params (M)']:.2f} M")

print("\n--- 🚨 중요 🚨 ---")
print("1. 'YourAutoencoder'와 'YourSiameseNetwork'는 예시 모델입니다.")
print("   논문의 결과로 사용하려면 반드시 사용자의 실제 모델 코드로 교체해야 합니다.")
print("2. 'EfficientNet-B0'를 실제 사용한 버전(예: B5 또는 B7)으로 변경하세요.")
print(f"3. 이 계산은 입력 크기 {input_size}를 기준으로 합니다. 이 크기가 실제 실험과 동일한지 꼭 확인하세요.")