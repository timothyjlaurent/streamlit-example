from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL.Image import Image

from fastai.learner import load_learner


"""
# Emotion Detector

Upload an image to predict the em0tion:

"""

MODEL_PATHS = [
    'models/resnet-34-facial-expressions.pkl'
]

@st.cache()
def load_fastai_classifier(path: str):
    return load_learner(path)


with st.echo(code_location='below'):
    model_name = st.selectbox("Choose model:", MODEL_PATHS)
    model = load_fastai_classifier(model_name, index=0)
    uploaded_file = st.file_uploader("Upload an image", type="file_type")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    model.predict(uploaded_file)
