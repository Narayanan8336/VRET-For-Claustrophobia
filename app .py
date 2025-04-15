from flask import Flask, request, render_template, jsonify, send_file
import torch
from PIL import Image
from io import BytesIO
from checkpoints import load_model, process_image  # Replace with your actual model and functions

app = Flask(__name__)

# Load models
model_ours = load_model('gmm_final.pth')
model_cp_vton = load_model('cp_vton_model.pth')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image_route():
    reference_image = request.files['reference_image']
    target_image = request.files['target_image']
    
    # Load images
    reference_image = Image.open(reference_image).convert('RGB')
    target_image = Image.open(target_image).convert('RGB')
    
    # Process images with models
    output_ours = process_image(model_ours, reference_image, target_image)
    output_cp_vton = process_image(model_cp_vton, reference_image, target_image)
    
    # Convert output to image and send as response
    output_ours_io = BytesIO()
    output_ours.save(output_ours_io, format='PNG')
    output_ours_io.seek(0)
    
    output_cp_vton_io = BytesIO()
    output_cp_vton.save(output_cp_vton_io, format='PNG')
    output_cp_vton_io.seek(0)
    
    return jsonify({
        'ours': output_ours_io.getvalue(),
        'cp_vton': output_cp_vton_io.getvalue()
    })

if __name__ == '__main__':
    app.run(debug=True)
