# Iris Flower Classifier (MLP with Gradio)

This repository implements a **Multilayer Perceptron (MLP)** using **TensorFlow/Keras** to classify iris flower species from the famous [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). The model is deployed using **Gradio** to provide an interactive web interface.

---

## Model Overview

- **Input Features**: Sepal Length, Sepal Width, Petal Length, Petal Width
- **Classes**:
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica
- **Architecture**: 
  - Input Layer (4 features)
  - Dense Layer (10 neurons, ReLU)
  - Dense Layer (8 neurons, ReLU)
  - Output Layer (3 neurons, Softmax)

Project Structure
mlp-iris-classifier/
│
├── app.py                # Main script with MLP and Gradio deployment
├── iris_mlp_model.h5     # Trained model (auto-generated on first run)
├── requirements.txt      # Dependencies
└── README.md             # This file
