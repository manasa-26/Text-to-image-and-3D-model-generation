import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cpu"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

prompt = "a photo of a bird sitting on a tree"

image = pipe(prompt,).images[0]  
    
image.save("Bird_on_tree.png")