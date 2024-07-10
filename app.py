from flask import Flask, render_template, request
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second_page', methods=['POST'])
def second_page():
    password = request.form['password']
    # Check if the password is correct
    if password == "FYP2024":
        return render_template('second_page.html')
    else:
        return "<span style='color: red; font-weight: bold; font-size: 32px; text-align: centre'>Incorrect password</span>"

@app.route('/project_selection')
def project_selection():
    return render_template('project_selection.html')    

@app.route('/about_team')
def about_team():
    return render_template('about_team.html')

@app.route('/3d_model_generation', methods=['GET', 'POST'])
def model_generation():
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')  # Get the prompt from the form data
        image_path = "static/img_0_0_0.png"  # Path to the regenerated image
    else:
        prompt = None  # Set prompt to None for GET requests
        image_path = "static/gen_0.png"  # Default path to the initial image

    return render_template('3d_model_generation.html', prompt=prompt, image_url=image_path)

@app.route('/text_to_image_synthesis', methods=['GET', 'POST'])
def text_to_image_synthesis():
    if request.method == 'POST':
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cpu"
        
        prompt = request.form['prompt']
        
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to(device)

        image = pipe(prompt).images[0]  
        image_path = "static/Bird_on_tree.png"
        image.save(image_path)  # Save the generated image
        
        return render_template('text_to_image_synthesis.html', prompt=prompt, image_url=image_path)
    else:
        return render_template('text_to_image_synthesis.html')

if __name__ == '__main__':
    app.run(debug=True)
