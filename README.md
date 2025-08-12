# **Hyperspectral Image Processing Project**

## **Project Overview**
This project aims to classify fruit ripeness levels using hyperspectral image processing techniques and machine learning algorithms. By extracting meaningful features from high-resolution hyperspectral images and applying machine learning models, the project provides an effective solution for automated classification in quality control and other fields where detailed image analysis is crucial.

## **Objectives**
1. **Feature Extraction and Dimensionality Reduction:**  
   Use Principal Component Analysis (PCA) and Competitive Adaptive Reweighted Sampling (CARS) to reduce data dimensionality and select the most informative bands for classification.
   
2. **Model Training and Evaluation:**  
   Train and evaluate machine learning models, including AlexNet (CNN), Decision Tree, and Random Forest, to classify ripeness levels with the highest accuracy.

3. **Comparison of Algorithms:**  
   Compare the effectiveness of SPA (Successive Projections Algorithm) and CARS for band selection and assess model performance before and after dimensionality reduction.

## **Key Techniques and Algorithms**

- **Principal Component Analysis (PCA):**  
  PCA is used to reduce data dimensions by selecting the most significant components from hyperspectral images, improving computational efficiency.

- **Competitive Adaptive Reweighted Sampling (CARS):**  
  CARS further refines the feature selection by identifying the most informative bands from those chosen by PCA, enhancing model performance.

- **Machine Learning Models:**
  - **AlexNet (CNN):** Applied to hyperspectral data for image classification, providing an end-to-end learning model.
  - **Decision Tree and Random Forest:** Supervised learning models trained with selected bands to classify ripeness levels.

## **Project Workflow**
1. **Data Preprocessing:**  
   Load and crop hyperspectral images to a consistent size, preparing them for analysis.

2. **Dimensionality Reduction:**  
   Apply PCA and CARS to reduce the dataset's dimensionality and select the most relevant bands, leading to more efficient model training.

3. **Model Training and Evaluation:**  
   Train various models (AlexNet, Decision Tree, Random Forest) and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.

4. **Performance Analysis and Visualization:**  
   Visualize the models' performance with confusion matrices and ROC curves to compare results before and after dimensionality reduction.






