# Alzheimer's Project

## Project Overview

This project focuses on the analysis of brain scan images for the detection and study of Alzheimer's disease. Utilizing a collection of brain scans, the goal is to identify patterns or markers associated with different stages of Alzheimer's, leveraging machine learning algorithms and image processing techniques. The project aims to contribute to the early detection and progression monitoring of Alzheimer's disease by categorizing the scans into distinct groups indicating the disease's severity.

## Dataset

The dataset for this project is a collection of brain scan images specifically designed for studying Alzheimer's disease. The scans are meticulously categorized into four groups based on the severity of the disease:
- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

Dataset source: https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset/data

### Characteristics
- The dataset includes a diverse set of brain scan images across the mentioned categories.
- Each image is preprocessed to ensure uniformity in size and resolution, facilitating more accurate analysis and model training.

## Requirements

To run this project, you'll need the following:
```plaintext
Python 3.8+ (The project was developed with Python 3.8 compatibility in mind.)
NumPy (For numerical computing and array operations.)
Pandas (For data manipulation and analysis.)
tqdm (For displaying progress bars during lengthy operations.)
scikit-learn (For machine learning model training, evaluation, and utilities like train/test split, confusion matrix, classification metrics, and class weight computation.)
TensorFlow 2.x (For building and training neural network models.)
Keras (Integrated within TensorFlow for model definition and training. Utilized for creating sequential and functional API models, as well as for various layers, callbacks, and optimizers.)
Keras Tuner (For hyperparameter tuning of Keras models.)
LIME (Local Interpretable Model-agnostic Explanations) for explaining predictions of image classifiers.
Matplotlib and seaborn (For data visualization and plotting.)

```

## Usage

To execute the analysis:
```plaintext
0. Download the dataset from Kaggle and save it in your working directory. 
1. Navigate to the 'alzheimers.ipynb' notebook within your Jupyter environment.
2. Sequentially run the cells, starting from data loading and preprocessing to training and evaluating the models.
```

## Results

I explored the performance of a Fully Connected Neural Network (FNN) and a hybrid model combining Residual Neural Network (ResNet) with CNN. 

The FNN displayed limitations in spatial feature extraction, leading to lesser accuracy. However, after implementing class weights to address the imbalance in the dataset, the accuracy of the FNN model significantly improved from 80% to 86%. This adjustment highlights the importance of considering dataset characteristics, such as imbalance, in model training to enhance performance.

The ResNet-CNN hybrid, while theoretically capable of deeper feature extraction due to its complex architecture, did not surpass the standalone CNN model in this specific analysis. It's important to note, however, that the ResNet-CNN model was not specifically tuned for this task. I used configurations optimized for the CNN model, suggesting that with appropriate tuning, the ResNet-CNN model has the potential to outperform the standalone CNN. This indicates a promising area for future research, where targeted optimizations of the ResNet-CNN model could unveil improvements in accurately diagnosing Alzheimer's disease stages.

Accuracy: 
FNN 86%
CNN 98%
ResNet- CNN 96%

## Explainability with LIME Explainer

In addition to developing models for the detection of Alzheimer's disease stages, this project also emphasizes the importance of model interpretability. I employed the LIME (Local Interpretable Model-agnostic Explanations) explainer to shed light on the decision-making process of the CNN model. The LIME explainer was particularly useful in generating explanations for individual predictions, and it allowed to visualize which features in the brain scan images were most influential in the model's classification decisions.


## Acknowledgments

Special thanks to the organizations and individuals who made the dataset available for research purposes. Their contributions are invaluable to advancements in understanding and detecting Alzheimer's disease.
