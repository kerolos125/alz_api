from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load model
model = tf.keras.models.load_model("model1.h5")

# Class labels
class_labels = ['Non Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # image(مواصفاتها)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((128, 128))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        result = class_labels[predicted_class]

        return JSONResponse(content={"prediction": result})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)