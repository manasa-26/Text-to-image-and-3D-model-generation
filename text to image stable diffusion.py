from transformers import pipeline, set_seed
import torch
from diffusers import StableDiffusionPipeline

class CFG:
    device = "cuda"  # Change to "cpu" if GPU is not available
    seed = 42
    generator = torch.Generator(device)  # Create torch.Generator for seeding
    generator.manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Initialize the StableDiffusionPipeline model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token='your_hugging_face_auth_token',  # Replace with your Hugging Face token
    guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

# Function to generate images based on a prompt
def generate_image(prompt, model):
    # Generate image using the model
    image = model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    # Resize image to specified size
    image = image.resize(CFG.image_gen_size)
    return image

# Example usage:
if __name__ == "__main__":
    set_seed(CFG.seed)  # Set seed for reproducibility
    prompt = "A serene landscape with mountains and a lake"  # Example prompt
    generated_image = generate_image(prompt, image_gen_model)
    generated_image.show()  # Display the generated image