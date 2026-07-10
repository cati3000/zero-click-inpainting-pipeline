import argparse
from pipeline import ZeroClickInpainter
from utils import download_image
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Run Zero-Click Inpainting")
    parser.add_argument("--url", type=str, required=True, help="URL of the input image")
    parser.add_argument("--detect", type=str, required=True, help="Object to detect and remove (e.g., 'human')")
    parser.add_argument("--target", type=str, required=True, help="Target prompt to replace the object with")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt")
    parser.add_argument("--output", type=str, default="result.png", help="Output filename")
    
    args = parser.parse_args()

    engine = ZeroClickInpainter()
    init_image = download_image(args.url)

    result, final_mask, bbox_log = engine.process(
        input_image=init_image,
        detect_prompt=args.detect,
        target_prompt=args.target,
        negative_prompt=args.negative,
        scale_mode="auto"
    )

    result.save(args.output)
    print(f"Success! Image saved to {args.output}")

if __name__ == "__main__":
    main()