import os
from typing import List
from fastapi import FastAPI, File, UploadFile
import numpy as np
from mlflow.keras import load_model
from PIL import Image
from datetime import datetime
import os


app = FastAPI()
model_path='/home/clifford/clifford/clifflearns/insti/CS5830/a05/mlartifacts/0/077185e74e6749c0890479ff9e473523/artifacts/model'

def load__model(path: str):
    """
    Load the Keras Sequential model from the specified path.

    Parameters:
        path (str): Path to the saved model file.

    Returns:
        Sequential: Loaded Keras Sequential model.
    """
    return load_model(path)

def predict_digit(model, data_point: List[float]) -> str:
    """
    Predict the digit using the loaded model.

    Parameters:
        model (Sequential): Keras Sequential model.
        data_point (list): Serialized array of 784 elements representing the image.

    Returns:
        str: Predicted digit.
    """
    # Preprocess data_point if needed (e.g., reshape)
    probs = model.predict(data_point)
    return str(np.argmax(probs))


def format_image(image_file):
    return image_file.resize((28, 28))



@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to predict the digit from an uploaded image.

    Parameters:
        file (UploadFile): Uploaded image file.
        model_path (str): Path to the saved model file.

    Returns:
        dict: Predicted digit.
    """
    # Save the uploaded image
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    contents = await file.read()
    with open(f"uploaded_images/uploaded_image_{formatted_datetime}.jpg", 'wb') as f:
        f.write(contents)

    # Load the model

    model = load__model(model_path)
    model.trainable=False
    # Convert image to serialized array of 784 elements
    # Perform image processing and convert to numpy array
    
    img = Image.open(f"/home/clifford/clifford/clifflearns/insti/CS5830/a06/uploaded_images/uploaded_image_{formatted_datetime}.jpg")
    img = img.convert('L')
    if img.size!=(28,28):
        img = format_image(img)
        img.save(f"/home/clifford/clifford/clifflearns/insti/CS5830/a06/uploaded_images/uploaded_image_{formatted_datetime}.jpg")
    img = Image.open(f"/home/clifford/clifford/clifflearns/insti/CS5830/a06/uploaded_images/uploaded_image_{formatted_datetime}.jpg")
    img_array = np.array(img)
    flattened_array = img_array.reshape(-1)
    flattened_array=flattened_array.astype('float32')
    normalized_array = flattened_array / 255.0  # Normalize pixel values
    normalized_array=normalized_array[None,:]
    #data_point=format image(file)

    # Predict the digit
    digit = predict_digit(model, normalized_array)

    return {"digit": digit}
