# Fruit-Ripeness-Detection-Project-with-Hyperspectral-Image-Processing
Hyperspectral Image Processing Project
Project Overview
This project aims to classify fruit ripeness levels using hyperspectral image processing techniques and machine learning algorithms. By extracting meaningful features from high-resolution hyperspectral images and applying machine learning models, the project provides an effective solution for automated classification in quality control and other fields where detailed image analysis is crucial.

Objectives
Feature Extraction and Dimensionality Reduction:
Use Principal Component Analysis (PCA) and Competitive Adaptive Reweighted Sampling (CARS) to reduce data dimensionality and select the most informative bands for classification.

Model Training and Evaluation:
Train and evaluate machine learning models, including AlexNet (CNN), Decision Tree, and Random Forest, to classify ripeness levels with the highest accuracy.

Comparison of Algorithms:
Compare the effectiveness of SPA (Successive Projections Algorithm) and CARS for band selection and assess model performance before and after dimensionality reduction.

Key Techniques and Algorithms
Principal Component Analysis (PCA):
PCA is used to reduce data dimensions by selecting the most significant components from hyperspectral images, improving computational efficiency.

Competitive Adaptive Reweighted Sampling (CARS):
CARS further refines the feature selection by identifying the most informative bands from those chosen by PCA, enhancing model performance.

Machine Learning Models:

AlexNet (CNN): Applied to hyperspectral data for image classification, providing an end-to-end learning model.
Decision Tree and Random Forest: Supervised learning models trained with selected bands to classify ripeness levels.
Project Workflow
Data Preprocessing:
Load and crop hyperspectral images to a consistent size, preparing them for analysis.

Dimensionality Reduction:
Apply PCA and CARS to reduce the dataset's dimensionality and select the most relevant bands, leading to more efficient model training.

Model Training and Evaluation:
Train various models (AlexNet, Decision Tree, Random Forest) and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.

Performance Analysis and Visualization:
Visualize the models' performance with confusion matrices and ROC curves to compare results before and after dimensionality reduction.

Folder Structure
plaintext
Kodu kopyala
.
├── data/                           # Directory containing hyperspectral images
├── models/                         # Directory for storing trained models
├── src/                            # Source code files
│   ├── preprocess.py               # Preprocessing scripts (e.g., PCA, CARS)
│   ├── train_model.py              # Model training and evaluation scripts
│   └── visualize.py                # Visualization scripts (confusion matrix, ROC curve)
├── README.md                       # Project description
└── requirements.txt                # Required libraries
Installation and Setup
Clone the repository:

bash
Kodu kopyala
git clone https://github.com/yourusername/hyperspectral-image-processing.git
cd hyperspectral-image-processing
Install dependencies: Install required libraries from requirements.txt:

bash
Kodu kopyala
pip install -r requirements.txt
Dataset Preparation:
Place your hyperspectral images in the data/ folder. The images should be in .hdr and .bin formats for compatibility with the provided code.

Running the Project
Data Preprocessing:

bash
Kodu kopyala
python src/preprocess.py
This script performs data loading, PCA, and CARS to prepare the images for model training.

Model Training and Evaluation:

bash
Kodu kopyala
python src/train_model.py
This script trains the models and evaluates their performance on the selected features.

Visualization:

bash
Kodu kopyala
python src/visualize.py
Use this script to visualize the confusion matrix and ROC curve for model evaluation.

Results
Accuracy and Performance Comparison:
Models were evaluated based on metrics such as accuracy, precision, recall, and F1-score. The Random Forest model, trained with bands selected by CARS, achieved the highest accuracy.

Execution Times:
The performance of SPA and CARS algorithms was compared based on execution times and band selection effectiveness, with CARS yielding higher classification accuracy.

Technologies Used
Programming Language: Python
Libraries: numpy, scikit-learn, tensorflow, spectral, matplotlib, seaborn
