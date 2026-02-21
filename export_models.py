
import os
import torch
import tensorrt as trt
import numpy as np
from ultralytics import YOLO
from vision.resnet_recognition import ResNetRecognizerV2

def export_yolo(path, imgsz=640):
    print(f"\n🚀 Exporting YOLO: {path} (imgsz={imgsz})")
    try:
        model = YOLO(path)
        # H100 supports FP16 (half=True)
        # simplify=True usually helps ONNX optimization
        out = model.export(format='engine', device=0, half=True, dynamic=False, imgsz=imgsz, simplify=True)
        print(f"✅ Success: {out}")
    except Exception as e:
        print(f"❌ Failed to export {path}: {e}")

def build_resnet_engine(onnx_path, engine_path, fp16=True):
    print(f"⚙️ Building TRT Engine from {onnx_path}...")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Create network definition with explicit batch setting
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # Enable FP16
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("   (FP16 Enabled)")
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"❌ ONNX Parse Error: {parser.get_error(error)}")
            return False
            
    # H100 optimization profile? No, standard settings should work.
    # Build serialized network
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("❌ Failed to build serialized engine.")
            return False
            
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"✅ Saved Engine: {engine_path}")
        return True
    except Exception as e:
        print(f"❌ Build Error: {e}")
        return False

def export_resnet(path):
    print(f"\n🚀 Exporting ResNet: {path}")
    onnx_path = path.replace('.pt', '.onnx')
    engine_path = path.replace('.pt', '.engine')
    
    if os.path.exists(engine_path):
        print(f"⏭️ Engine already exists: {engine_path}")
        # return # Force rebuild? User asked to export.
    
    try:
        # Load model wrapper
        recognizer = ResNetRecognizerV2(weights_path=path, device='cuda')
        model = recognizer.model
        model.eval()
        
        # Dummy Input (Batch 1, 3, 128, 128) - standard JNR size
        dummy_input = torch.randn(1, 3, 128, 128, device='cuda')
        
        # Export to ONNX
        print("   -> Exporting to ONNX...")
        torch.onnx.export(model, dummy_input, onnx_path,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                          opset_version=12)
        print(f"   -> ONNX saved: {onnx_path}")
        
        # Build TRT
        build_resnet_engine(onnx_path, engine_path, fp16=True)
        
    except Exception as e:
        print(f"❌ Failed ResNet export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. Export YOLO models (Player, Ball, Pitch)
    # Pitch might utilize larger resolution? Config says 640.
    export_yolo("models/yolo_player.pt", imgsz=640)
    export_yolo("models/yolo_ball.pt", imgsz=640)
    
    # Pitch keypoint model often uses 640 too
    export_yolo("models/yolo_pitch.pt", imgsz=640)
    
    # 2. Export ResNet
    export_resnet("models/resnet34_rgb_jnr.pt")
