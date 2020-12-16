from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image

from fastai.learner import load_learner


"""
# Emotion Detector

Upload an image to predict the em0tion:

"""

MODEL_PATHS = [
    'models/resnet-34-facial-expressions.pkl'
]

@st.cache
def load_fastai_classifier(path: str):
    model = load_learner(path)
    return model


with st.echo(code_location='below'):
    model_name = st.selectbox("Choose model:", MODEL_PATHS)
    model = load_learner(model_name)
    uploaded_file = st.file_uploader("Upload an image", type="file_type")
    uploaded_file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    model.predict(uploaded_file)
