from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.model import load_dog_breed_model, predict_breed, get_breed_name, preprocess_image
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_dog_breed_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
   # Read the uploaded file contents as bytes
    contents = await file.read()
        
    # Convert the bytes data to a PIL image
    pil_image = Image.open(io.BytesIO(contents))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image_tensor = tf.convert_to_tensor(pil_image, dtype=tf.float32)

    processed_image = preprocess_image(image_tensor)
    predicted_class = predict_breed(model, processed_image)
    breed_name = get_breed_name(predicted_class)
    return JSONResponse(content={"predicted_class": breed_name})