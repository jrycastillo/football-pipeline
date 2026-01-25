
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import json

def test_crop(model_path, crop_path):
    print(f"--- Testing {model_path} ---")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    img = cv2.imread(crop_path)
    h, w = img.shape[:2]
    
    # Test variants
    variants = {
        "Raw": img,
        "Upscale_2x": cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4),
        "Upscale_448px": cv2.resize(img, (int(w*(448/h)), 448), interpolation=cv2.INTER_LANCZOS4),
        "Sharpened_448": None # Will add
    }
    
    # Sharpened variant
    up = variants["Upscale_448px"]
    gauss = cv2.GaussianBlur(up, (0,0), 3)
    variants["Sharpened_448"] = cv2.addWeighted(up, 1.5, gauss, -0.5, 0)

    for name, v in variants.items():
        if v is None: continue
        pil = Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": "Identify the jersey number on this player. Return JSON: {\"number\": int}"}
            ]
        }]]
        
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=32)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        print(f"[{name}] Result: {output_text}")

if __name__ == "__main__":
    crop_p = "/home/ubuntu/football/output/clipped_ikorudo_tornadoes/crops/390.jpg"
    test_crop("Qwen/Qwen2.5-VL-3B-Instruct", crop_p)
    # test_crop("/home/ubuntu/football/runs/qwen_h100_finetune_merged", crop_p)
