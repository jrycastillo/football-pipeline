from transformers import SiglipImageProcessor, SiglipVisionModel
import torch
from PIL import Image
import numpy as np

model_id = "google/siglip-base-patch16-224"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model {model_id} on {device}...")
processor = SiglipImageProcessor.from_pretrained(model_id)
model = SiglipVisionModel.from_pretrained(model_id).to(device)
model.eval()

# Dummy image
dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

with torch.no_grad():
    inputs = processor(images=dummy_img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # The pooler_output is the representative embedding for the image
    image_embeds = outputs.pooler_output
    print(f"Embedding shape: {image_embeds.shape}")

print("SigLIP Test Success!")
