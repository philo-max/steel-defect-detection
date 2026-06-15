import argparse
import os
import subprocess
from pathlib import Path
from loguru import logger
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 PyTorch to ONNX Export & Slimming")
    parser.add_argument("--model", default="models/weights/steel_defect.pt", help="PyTorch model weights path")
    parser.add_argument("--img-size", type=int, default=640, help="Export input image size (default: 640)")
    parser.add_argument("--device", default="cpu", help="Device to use for export (default: cpu)")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model}")
        return

    logger.info(f"Loading PyTorch model from {model_path}...")
    model = YOLO(str(model_path))

    logger.info("Exporting to ONNX format...")
    # Export the model
    # dynamic=True is optional but for batch size dynamic is useful. YOLOv8 supports dynamic shape export.
    exported_path_str = model.export(
        format="onnx",
        imgsz=args.img_size,
        device=args.device,
        dynamic=True,
        simplify=True,
        verbose=True
    )
    
    # Locate the exported ONNX file
    exported_onnx_path = model_path.with_suffix(".onnx")
    if not exported_onnx_path.exists():
        # Sometimes it's written in the directory of export
        exported_onnx_path = Path(exported_path_str)
        
    if exported_onnx_path.exists():
        logger.info(f"Successfully exported ONNX model to: {exported_onnx_path}")
        
        # Try running onnxslim if available
        try:
            logger.info("Attempting to run onnxslim graph optimization...")
            slim_onnx_path = exported_onnx_path.parent / f"{exported_onnx_path.stem}_slim.onnx"
            # run command: onnxslim <input> <output>
            proc = subprocess.run(["onnxslim", str(exported_onnx_path), str(slim_onnx_path)], capture_output=True, text=True)
            if proc.returncode == 0 and slim_onnx_path.exists():
                logger.info(f"Successfully slimmed ONNX model to: {slim_onnx_path}")
                # Replace the original onnx with the slim one
                os.remove(exported_onnx_path)
                os.rename(slim_onnx_path, exported_onnx_path)
                logger.info("Replaced original ONNX model with optimized version.")
            else:
                logger.warning(f"onnxslim execution failed: {proc.stderr or 'File not created'}")
        except Exception as e:
            logger.warning(f"Could not run onnxslim: {e}. Keeping default exported ONNX model.")
    else:
        logger.error("Failed to find exported ONNX model file.")

if __name__ == "__main__":
    main()
