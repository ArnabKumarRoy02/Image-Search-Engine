# Reverse Image Search Engine: Understanding Embeddings

This project implements a Reverse Image Search Engine that leverages embedding techniques on the Caltech101 dataset. The goal of the system is to allow users to search for visually similar images based on a query image. The embedding approach enables the system to capture the visual similarity between images, even if they belong to different categories.

## Features

- Indexing: The dataset is indexed to create embeddings for each image using deep learning techniques.
- Querying: Users can input an image as a query and retrieve visually similar images from the dataset.
- Embedding Visualization: The project includes visualization techniques to explore and understand the image embeddings.

## Dataset

This project uses the [Caltech101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) which contains 101 categories of objects, with 40 to 800 images per category. The dataset is split into training and testing sets, with 30 and 50 images per category respectively. The images are of variable sizes and aspect ratios.

## Installation

To run the Reverse Image Search Engine, follow the steps:

1. Clone the repository

```shell
git clone https://github.com/ArnabKumarRoy02/Image-Search-Engine.git
```

2. Create a virtual environment

```shell
cd Image-Search-Engine
conda create -n env
conda activate env
conda install python==3.8.16
```

3. Install the dependencies

```shell
pip install -r requirements.txt
```

4. Launch the flask app

```shell
python app.py
```

## Usage

1. Access the Reverse Image Search Engine by opening the provided URL in a web browser.
2. Upload an image or provide the URL of an image as a query.
3. Submit the query and wait for the system to retrieve visually similar images.
4. Explore the search results and interact with the visualization features to gain insights into the embeddings.

## Contributing

Contributions are welcome! If you want to enhance the Reverse Image Search Engine, submit a pull request with your proposed changes. Please follow the existing code style and include appropriate tests.

## License

This project is licensed under the [MIT License](LICENSE)

## References

- [Caltech101 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)
- [Caltech101 Dataset Paper](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.pdf)
- [Practical Deep Learning for Cloud, Mobile and Edge](https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/)