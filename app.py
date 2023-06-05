from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import Resnet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = Resnet50(weights='imagenet', include_top=False, pooling='avg')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for image upload and search
@app.route('/search', methods=['POST'])
def search():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No image file selected')
    
    file = request.files['file']
    
    # Check if the file is a valid image
    if file.filename == '':
        return render_template('index.html', error='No image file selected')
    
    try:
        img = Image.open(file)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Resize to ResNet50 input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features using ResNet50 model
        query_features = model.predict(img_array)
        
        # Calculate similarity scores
        similarities = []
        for dataset_image_features in dataset_features:
            similarity = cosine_similarity(query_features, dataset_image_features)
            similarities.append(similarity)
        
        # Sort images based on similarity scores
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_images = [dataset_images[idx] for idx in sorted_indices]
        
        return render_template('results.html', query_image=file, similar_images=sorted_images)
    
    except Exception as e:
        return render_template('index.html', error='Error processing image: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
