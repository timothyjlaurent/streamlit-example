from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import PIL

from fastai.learner import load_learner
import torchvision.transforms as tfms
from fastai.vision import *
from fastai.vision.core import PILImage
import numpy as np
import plotly.graph_objects as go


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
    uploaded_file = st.file_uploader("Upload an image")
    if uploaded_file:
        image = PIL.Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        predictions = model.predict(PILImage(image))
        f"Predicted Emotion: **{predictions[0]}**"
        df = pd.DataFrame([predictions[2].numpy()], columns=model.dls.vocab)
        df
        fig = go.Figure([go.Bar(x=list(model.dls.vocab), y=predictions[2].numpy())])
        st.plotly_chart(
            fig,
            use_container_width=True
        )
