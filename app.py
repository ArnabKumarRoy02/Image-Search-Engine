import pickle
import random
import matplotlib
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained model and data
with open('data/model-finetuned.h5', 'rb') as f:
    model = pickle.load(f)

with open('data/features-caltech101-resnet.pickle', 'rb') as f:
    dataset_features = pickle.load(f)

with open('data/filenames-caltech101.pickle', 'rb') as f:
    dataset_filenames = pickle.load(f)

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
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

        # Extract features using ResNet50 model
        query_features = model.predict(img_array)

        # Calculate similarity scores
        similarities = []
        for dataset_image_features in dataset_features:
            similarity = np.dot(query_features, dataset_image_features.T)
            similarities.append(similarity)

        # Sort images based on similarity scores
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_filenames = [dataset_filenames[idx] for idx in sorted_indices]

        return render_template('results.html', query_image=file, similar_filenames=sorted_filenames)

    except Exception as e:
        return render_template('index.html', error='Error processing image: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
