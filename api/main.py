# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:26:13 2022

@author: vishnu
"""
from fastapi import FastAPI, File, UploadFile
import uvicorn 
#import asyncio
import nest_asyncio
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/tf_final_model")
CLASS_NAMES = ['Tomato_Bacterial_spot',
                'Tomato_Early_blight',
                'Tomato_Late_blight',
                'Tomato_Leaf_Mold',
                'Tomato_Septoria_leaf_spot',
                'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot',
                'Tomato__Tomato_YellowLeaf__Curl_Virus',
                'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy']

@app.get("/ping")
async def ping():
    return "Hello, changed 123"

''' 
    uncomment the below code if encounter the below error.
    once executed, for next runs, the below code can be commented.
'''
## RuntimeError: asyncio.run() cannot be called from a running event loop Error
# from unsync import unsync
# @unsync
# async def example_async_function():
#     await asyncio.sleep(0.1)
#     return "Run Successfully!"
# print(example_async_function().result())

# async def main():
#     if __name__ == "__main__":
#         nest_asyncio.apply()
#         uvicorn.run(app, host = 'localhost', port=8000)

# asyncio.run(main())


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
    ):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    prediction = MODEL.predict(img_batch)
    print(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    print(predicted_class)
    confidence = np.max(prediction[0]) * 100
    return {
            'class':predicted_class,
            'confidence':round(float(confidence),2)
        }

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host = 'localhost', port=8000)