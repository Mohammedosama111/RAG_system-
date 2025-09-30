# Insect Recognition Project

This repository contains Jupyter notebooks for an insect recognition project, demonstrating different approaches to classify insect images using deep learning models.

## Project Overview

The primary goal of this project is to develop and evaluate machine learning models capable of accurately classifying images of various insect species. The project explores different convolutional neural network (CNN) architectures, including a custom CNN model and a transfer learning approach using the VGG19 model.

## Dataset

The dataset used for this project is named `Insects Recognition.zip`. It is expected to be located in a Google Drive directory and is extracted to a local path (e.g., `/content/dataset` or `/content/Insects Recognition`) within the Colab environment.

The dataset comprises images of the following five insect classes:
- Butterfly
- Dragonfly
- Grasshopper
- Ladybird
- Mosquito

## Notebooks

This repository includes the following Jupyter notebooks, each detailing a specific aspect of the insect recognition task:

### 1. `CNN_insects_recognition.ipynb`

This notebook implements a custom Convolutional Neural Network (CNN) for insect image classification. It covers data loading, preprocessing, model definition, training, and evaluation.

**Key Features:**
- Data loading from Google Drive.
- Image data augmentation using `ImageDataGenerator` (rescaling, rotation, horizontal flip, fill mode).
- Custom CNN architecture with `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten`, `Dense`, and `BatchNormalization` layers.
- Model compilation with Adam optimizer and `categorical_crossentropy` loss.
- Training with `EarlyStopping` and `ReduceLROnPlateau` callbacks.

### 2. `Insect_Recognition_VGG19.ipynb`

This notebook demonstrates the application of transfer learning using the pre-trained VGG19 model for insect recognition. It leverages the powerful features learned by VGG19 on a large dataset (ImageNet) and fine-tunes it for the specific task of insect classification.

**Key Features:**
- Utilization of the VGG19 model pre-trained on ImageNet, with its convolutional base frozen.
- Addition of custom classification layers on top of the VGG19 base, including `GlobalAveragePooling2D`, `Dropout`, and `Dense` layers.
- Data preprocessing and augmentation tailored for VGG19's input requirements (224x224 target size).
- Model training with `ModelCheckpoint` to save the best performing model.
- Splitting the dataset into training and testing sets (80% train, 20% test) programmatically.

### 3. `InsectsRecognition_final_ML__(1).ipynb`

This notebook appears to be a final or consolidated version, potentially incorporating elements from the other notebooks or exploring additional machine learning techniques. It includes data loading, splitting, and some preprocessing steps.

**Key Features:**
- Data loading and extraction from a ZIP file.
- Programmatic splitting of the dataset into training and testing sets (80% train, 20% test).
- Image resizing (e.g., to 128x128 pixels) during data loading.
- Counting and summing images across different categories to understand dataset distribution.
- Mentions of 


preprocessing steps like edge detection, although the specific implementation details for the ML model are not explicitly detailed in the provided snippets.

## Setup and Usage

To run these notebooks, you will need to:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Download the dataset:** The `Insects Recognition.zip` dataset needs to be available in your Google Drive. Ensure it's accessible from your Colab environment.
3.  **Open in Google Colab:** Upload and open the `.ipynb` files in Google Colab.
4.  **Run the cells:** Execute the cells sequentially. The notebooks handle mounting Google Drive and extracting the dataset.

## Dependencies

The notebooks primarily use the following Python libraries:

-   `tensorflow` and `keras` for building and training neural networks.
-   `numpy` for numerical operations.
-   `pandas` for data manipulation.
-   `matplotlib` for plotting and visualization.
-   `Pillow` (PIL) for image processing.
-   `opencv-python` (cv2) for image loading and manipulation.
-   `scipy` for scientific computing.
-   `scikit-learn` for machine learning utilities (though not explicitly shown in snippets, often used for metrics).

## Results and Evaluation

Each notebook includes training and validation metrics (e.g., accuracy, loss) during the model training phase. The `Insect_Recognition_VGG19.ipynb` specifically saves the best performing model based on validation accuracy.

## Future Work

Potential future enhancements for this project could include:

-   Exploring more advanced CNN architectures (e.g., ResNet, Inception).
-   Implementing more sophisticated data augmentation techniques.
-   Experimenting with different optimizers and learning rate schedules.
-   Performing detailed error analysis and confusion matrix generation.
-   Deploying the trained models as a web service or mobile application.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Author

Manus AI


