import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

## WIP
@st.cache_resource
def load_pipelines():
    pipe1 = pipeline("image-classification", model="dima806/crime_cctv_image_detection", device=0)
    pipe2 = pipeline("image-classification",model="dima806/crime_type_cctv_image_detection",device=0,)
    return pipe1, pipe2

pipe1, pipe2 = load_pipelines()


def pipe(image, pipe1, pipe2):
    res1 = pipe1(image)

    crime_score = 0
    for r in res1:
        if r["label"] == "Crime":
            crime_score = r["score"]
            break
    
    if crime_score > 0.5:
        res2 = pipe2(image)
        predicted_label = res2[0]["label"] if len(res2) > 0 else "Unknown"
        return crime_score, predicted_label
    
    return crime_score, "Normal"


st.title("Real-time Crime Detection")
## WIP


