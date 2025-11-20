"""
딥러닝 모델 FLOPS 측정 - thop 라이브러리 사용
Autoencoder, EfficientNet, Siamese Network, MobileNetV3
"""

import torch
import torch.nn as nn
from torchvision import models
from thop import profile, clever_format
import time

# ====================================================================
# 1. 설치 필요
# ====================================================================
"""
pip install thop
pip install torchvision
"""

# ====================================================================
# 2. Autoencoder 예제
# ====================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=64):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def measure_autoencoder_flops():
    """Autoencoder FLOPS 측정"""
    print("="*60)
    print("Autoencoder FLOPS 측정")
    print("="*60)
    
    model = Autoencoder(input_dim=784, encoding_dim=64)
    model.eval()
    
    # 입력 데이터 (batch_size=1, input_dim=784)
    input_tensor = torch.randn(1, 784)
    
    # FLOPS 측정
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    print()


# ====================================================================
# 3. EfficientNet 예제
# ====================================================================
def measure_efficientnet_flops():
    """EfficientNet FLOPS 측정"""
    print("="*60)
    print("EfficientNet-B0 FLOPS 측정")
    print("="*60)
    
    # EfficientNet-B0 모델 로드
    model = models.efficientnet_b0(pretrained=False)
    model.eval()
    
    # 입력 데이터 (batch_size=1, channels=3, height=224, width=224)
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # FLOPS 측정
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    print()


# ====================================================================
# 4. Siamese Network 예제
# ====================================================================
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
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 256),
        )
    
    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def measure_siamese_flops():
    """Siamese Network FLOPS 측정"""
    print("="*60)
    print("Siamese Network FLOPS 측정")
    print("="*60)
    
    model = SiameseNetwork()
    model.eval()
    
    # 입력 데이터 (이미지 쌍)
    input1 = torch.randn(1, 3, 105, 105)
    input2 = torch.randn(1, 3, 105, 105)
    
    # FLOPS 측정 (단일 입력 기준)
    flops, params = profile(model.cnn, inputs=(input1,), verbose=False)
    
    # Siamese는 같은 네트워크를 2번 사용하므로 x2
    total_flops = flops * 2
    
    # FC layer 추가
    fc_input = torch.randn(1, 256 * 6 * 6)
    fc_flops, fc_params = profile(model.fc, inputs=(fc_input,), verbose=False)
    total_flops += fc_flops * 2
    
    total_flops, params = clever_format([total_flops, params], "%.3f")
    
    print(f"FLOPs (전체): {total_flops}")
    print(f"Parameters: {params}")
    print()


# ====================================================================
# 5. MobileNetV3 예제
# ====================================================================
def measure_mobilenetv3_flops():
    """MobileNetV3 FLOPS 측정"""
    print("="*60)
    print("MobileNetV3-Large FLOPS 측정")
    print("="*60)
    
    # MobileNetV3-Large 모델 로드
    model = models.mobilenet_v3_large(pretrained=False)
    model.eval()
    
    # 입력 데이터
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # FLOPS 측정
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    print()
    
    # MobileNetV3-Small도 측정
    print("="*60)
    print("MobileNetV3-Small FLOPS 측정")
    print("="*60)
    
    model_small = models.mobilenet_v3_small(pretrained=False)
    model_small.eval()
    
    flops, params = profile(model_small, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")
    print()


# ====================================================================
# 6. 모든 모델 비교
# ====================================================================
def compare_all_models():
    """모든 모델의 FLOPS 비교"""
    print("\n" + "="*80)
    print(" "*20 + "딥러닝 모델 FLOPS 비교")
    print("="*80)
    
    models_info = []
    
    # Autoencoder
    model = Autoencoder()
    input_tensor = torch.randn(1, 784)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    models_info.append(("Autoencoder", flops, params))
    
    # EfficientNet-B0
    model = models.efficientnet_b0(pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    models_info.append(("EfficientNet-B0", flops, params))
    
    # Siamese Network (근사치)
    model = SiameseNetwork()
    input1 = torch.randn(1, 3, 105, 105)
    flops_cnn, _ = profile(model.cnn, inputs=(input1,), verbose=False)
    fc_input = torch.randn(1, 256 * 6 * 6)
    flops_fc, params = profile(model.fc, inputs=(fc_input,), verbose=False)
    total_flops = (flops_cnn + flops_fc) * 2
    models_info.append(("Siamese Network", total_flops, params))
    
    # MobileNetV3-Large
    model = models.mobilenet_v3_large(pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    models_info.append(("MobileNetV3-Large", flops, params))
    
    # MobileNetV3-Small
    model = models.mobilenet_v3_small(pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    models_info.append(("MobileNetV3-Small", flops, params))
    
    # 테이블 출력
    print(f"\n{'Model':<25} {'FLOPs':<20} {'Parameters':<20} {'GFLOPS':<15}")
    print("-"*80)
    
    for name, flops, params in models_info:
        flops_str, params_str = clever_format([flops, params], "%.3f")
        gflops = flops / 1e9
        print(f"{name:<25} {flops_str:<20} {params_str:<20} {gflops:<15.4f}")
    
    print("="*80)


# ====================================================================
# 7. 실제 추론 시간 측정 함수
# ====================================================================
def measure_inference_time(model, input_tensor, num_runs=100):
    """실제 추론 시간 측정"""
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # 실제 측정
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / num_runs
    return avg_time * 1000  # ms로 변환


def compare_with_inference_time():
    """FLOPS와 실제 추론 시간 비교"""
    print("\n" + "="*80)
    print(" "*15 + "FLOPS vs 실제 추론 시간 비교")
    print("="*80)
    
    print(f"\n{'Model':<25} {'FLOPs':<15} {'GFLOPS':<12} {'Inference Time (ms)':<20}")
    print("-"*80)
    
    # EfficientNet-B0
    model = models.efficientnet_b0(pretrained=False)
    input_tensor = torch.randn(1, 3, 224, 224)
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    inference_time = measure_inference_time(model, input_tensor)
    flops_str, _ = clever_format([flops], "%.3f")
    gflops = flops / 1e9
    print(f"{'EfficientNet-B0':<25} {flops_str[0]:<15} {gflops:<12.4f} {inference_time:<20.2f}")
    
    # MobileNetV3-Large
    model = models.mobilenet_v3_large(pretrained=False)
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    inference_time = measure_inference_time(model, input_tensor)
    flops_str, _ = clever_format([flops], "%.3f")
    gflops = flops / 1e9
    print(f"{'MobileNetV3-Large':<25} {flops_str[0]:<15} {gflops:<12.4f} {inference_time:<20.2f}")
    
    # MobileNetV3-Small
    model = models.mobilenet_v3_small(pretrained=False)
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    inference_time = measure_inference_time(model, input_tensor)
    flops_str, _ = clever_format([flops], "%.3f")
    gflops = flops / 1e9
    print(f"{'MobileNetV3-Small':<25} {flops_str[0]:<15} {gflops:<12.4f} {inference_time:<20.2f}")
    
    print("="*80)
    print("※ FLOPS가 낮을수록 계산량이 적고, 일반적으로 추론 시간도 짧습니다.")
    print("※ 실제 추론 시간은 하드웨어, 메모리 접근 패턴 등에 영향을 받습니다.")


# ====================================================================
# 8. 메인 실행
# ====================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*20 + "딥러닝 모델 FLOPS 측정 시작")
    print("="*80 + "\n")
    
    # 개별 모델 측정
    measure_autoencoder_flops()
    measure_efficientnet_flops()
    measure_siamese_flops()
    measure_mobilenetv3_flops()
    
    # 전체 비교
    compare_all_models()
    
    # 실제 추론 시간과 비교
    compare_with_inference_time()
    
    print("\n" + "="*80)
    print(" "*25 + "측정 완료!")
    print("="*80)