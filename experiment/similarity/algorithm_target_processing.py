
from pathlib import Path
from PIL import Image
from rembg import remove
from rembg.session_factory import new_session
from tqdm import tqdm

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

INPUT_BASE_DIR = Path(config.TARGET_IMAGE_SAMPLE_DIR)
OUTPUT_BASE_DIR = Path(config.TARGET_IMAGE_ALGORITHM_DIR)
IMAGE_PATTERNS = config.IMAGE_EXTENSIONS


def preprocess_image_for_algorithm(
    input_path: Path, 
    output_path: Path, 
    session
):
     
    try:
        # 1. 이미지 로드 및 표준화
        image = Image.open(input_path)
        if image.format == "GIF":
            image.seek(0)
        
        image_rgb = image.convert("RGB") 

        # 2. 배경 제거 (전달받은 GPU/CPU 세션 사용)
        image_nobg = remove(image_rgb, session=session)

        # 3. 내용물 기준으로 자르기 (Crop)
        bbox = image_nobg.getbbox()
        if not bbox:
            print(f"Skipping empty image: {input_path.name}")
            return
        
        # 4. 타이트하게 크롭된 이미지를 그대로 사용
        cropped_image = image_nobg.crop(bbox)

        # 5. 최종 저장 (PNG)
        # 크롭된 이미지를 리사이즈나 패딩 없이 바로 저장
        cropped_image.save(output_path, "PNG")

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}", file=sys.stderr)

# --- 3. 메인 실행 로직 ---

def main():
    print(f"Start processing for Target Images (Algorithm Standard)...")
    print(f"Input directory:  {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("-" * 30)

    # --- GPU 세션 생성 (실패 시 CPU로 폴백) ---
    print("Initializing rembg session (Attempting GPU)...")
    try:
        
        session = new_session(providers=['CUDAExecutionProvider'])
        print(" GPU session (CUDAExecutionProvider) initialized successfully.")
    except Exception as e:
        print(f"--- GPU session failed! Error: {e} ---", file=sys.stderr)
        print("Falling back to CPU... (This may be very slow)")
        try:
            # GPU 실패 시 CPU로 폴백
            session = new_session(providers=['CPUExecutionProvider'])
            print(" CPU session initialized.")
        except Exception as cpu_e:
            print(f"Fatal: Could not initialize CPU session: {cpu_e}", file=sys.stderr)
            return
    # --- --------------------------- ---
    print("-" * 30)

    # 출력 폴더 생성
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 입력 폴더에서 이미지 파일 스캔
    image_files = []
    for pattern in IMAGE_PATTERNS:
        image_files.extend(INPUT_BASE_DIR.glob(pattern))
    
    if not image_files:
        print(f"No images found in {INPUT_BASE_DIR}")
        return

    print(f"Found {len(image_files)} images to process.")

    # tqdm으로 진행 상황 표시
    for input_file in tqdm(image_files, desc="Processing Targets", unit="img"):
        # 저장 파일명: (원본파일이름).png
        output_file = OUTPUT_BASE_DIR / (input_file.stem + ".png")
        
        # 이미 처리된 파일은 건너뛰기
        if output_file.exists():
            continue
        
        # 전처리 함수 호출 (세션 전달)
        preprocess_image_for_algorithm(input_file, output_file, session)

    print("\nAll target processing complete.")

if __name__ == "__main__":
    main()