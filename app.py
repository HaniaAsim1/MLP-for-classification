# app.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import gradio as gr

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build and train the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

# Save the model
model.save("iris_mlp_model.h5")

# Gradio Interface
class_names = iris.target_names.tolist()

def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    loaded_model = load_model("iris_mlp_model.h5")
    prediction = loaded_model.predict(input_scaled)
    class_index = np.argmax(prediction)
    return f"Predicted Iris Type: {class_names[class_index]}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)"),
    ],
    outputs="text",
    title="Iris Flower Classifier (MLP)",
    description="Enter measurements to predict the Iris species using a trained MLP model."
)

iface.launch()
