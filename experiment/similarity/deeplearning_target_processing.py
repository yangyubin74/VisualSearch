from pathlib import Path
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session
from tqdm import tqdm
from typing import Optional

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# --- 1. 설정: 경로 정의 ---
INPUT_BASE_DIR = Path(config.TARGET_IMAGE_SAMPLE_DIR)
OUTPUT_BASE_DIR = Path(config.TARGET_IMAGE_DEEPLEARNING_DIR)
IMAGE_PATTERNS = config.IMAGE_EXTENSIONS


def preprocess_and_save(
    input_path: Path,
    output_path: Path,
    target_size: int = 512,
    session: Optional[object] = None
) -> str:
    """
    Target 이미지를 학습 데이터와 동일한 방식으로 전처리하고 저장
    
    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        target_size: 출력 이미지 크기 (기본값: 512)
        session: rembg GPU 세션 (옵션)
    
    Returns:
        str: 저장된 이미지 경로
    
    Raises:
        ValueError: 배경 제거 후 이미지가 비어있을 때
        Exception: 기타 처리 오류
    """
    try:
        # 출력 디렉토리가 없으면 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. 이미지 로드 및 표준화
        image = Image.open(input_path)
        if image.format == "GIF":
            image.seek(0)
        image_rgb = image.convert("RGB")
        
        # 2. 배경 제거 (전달받은 GPU 세션 사용)
        if session:
            image_nobg = remove(image_rgb, session=session)
        else:
            image_nobg = remove(image_rgb)
        
        # 3. 내용물 기준으로 자르기 (Crop)
        bbox = image_nobg.getbbox()
        if not bbox:
            raise ValueError(f"Empty image after background removal")
        
        cropped_image = image_nobg.crop(bbox)
        
        # 4. 크기 조절(Resize) 및 패딩(Padding)
        canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
        cropped_image.thumbnail((target_size, target_size))
        x_offset = (target_size - cropped_image.width) // 2
        y_offset = (target_size - cropped_image.height) // 2
        canvas.paste(cropped_image, (x_offset, y_offset))
        
        # 5. RGBA -> RGB 변환 (투명 배경을 흰색으로)
        rgb_canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        rgb_canvas.paste(canvas, mask=canvas.split()[3])  # Alpha 채널을 마스크로 사용
        
        # 6. 최종 저장 (PNG)
        rgb_canvas.save(output_path, "PNG")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error processing {input_path.name}: {e}", file=sys.stderr)
        raise


def main():
    print(f"Start processing for Target Images (Algorithm Standard)...")
    print(f"Input directory:  {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("-" * 50)

    # --- GPU 세션 생성 (실패 시 CPU로 폴백) ---
    print("Initializing rembg session (Attempting GPU)...")
    try:
        session = new_session(providers=['CUDAExecutionProvider'])
        print("✅ GPU session (CUDAExecutionProvider) initialized successfully.")
    except Exception as e:
        print(f"🚨 GPU session failed: {e}", file=sys.stderr)
        print("⚠️  Falling back to CPU... (This may be slower)")
        try:
            session = new_session(providers=['CPUExecutionProvider'])
            print("✅ CPU session initialized.")
        except Exception as cpu_e:
            print(f"❌ Fatal: Could not initialize CPU session: {cpu_e}", file=sys.stderr)
            return
    
    print("-" * 50)

    # 출력 폴더 생성
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 입력 폴더에서 이미지 파일 스캔 (중복 제거)
    image_files = []
    seen_files = set()
    for pattern in IMAGE_PATTERNS:
        for file in INPUT_BASE_DIR.glob(pattern):
            if file not in seen_files:
                image_files.append(file)
                seen_files.add(file)
    
    if not image_files:
        print(f"⚠️  No images found in {INPUT_BASE_DIR}")
        return

    print(f"Found {len(image_files)} images to process.\n")

    # 통계 변수
    success_count = 0
    skip_count = 0
    error_count = 0

    # tqdm으로 진행 상황 표시
    for input_file in tqdm(image_files, desc="Processing Targets", unit="img"):
        output_file = OUTPUT_BASE_DIR / (input_file.stem + ".png")
        
        # 이미 처리된 파일은 건너뛰기
        if output_file.exists():
            skip_count += 1
            continue
        
        # 전처리 함수 호출 (🔧 수정: target_size 명시적 전달)
        try:
            preprocess_and_save(
                input_path=input_file,
                output_path=output_file,
                target_size=512,
                session=session
            )
            success_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"⚠️  Skipping {input_file.name}: {str(e)[:60]}")

    # 최종 통계 출력
    print("\n" + "=" * 50)
    print("Processing Complete!")
    print(f"  ✅ Success: {success_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  ❌ Errors:  {error_count}")
    print(f"  📊 Total:   {len(image_files)}")
    print("=" * 50)


if __name__ == "__main__":
    main()