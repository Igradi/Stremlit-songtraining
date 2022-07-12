import plotly.express as px
import streamlit.components.v1 as components
import pandas as pd
from requests import head
import streamlit as st
import sklearn
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
st.set_page_config(page_title="Song Recommendation", layout="wide")

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


@st.cache
def get_data(filename):

    song_data = pd.read_csv(filename)

    return song_data


with header:
    st.title('Dobrodosli u projekt')

with dataset:
    song_data = get_data('data/song_data.csv')
    st.write(song_data.head())

with modelTraining:
    st.header('Time to train the model')
    st.text('Choose parameterers')
    duration = modelTraining.slider('Trajanje u ms', min_value=1,
                                    max_value=1800000, value=1)
    acousticness = modelTraining.slider(
        'Akusticnost', min_value=0, max_value=1, value=0)
    danceability = modelTraining.slider(
        'Plesnost', min_value=10, max_value=20, value=15)
    energy = modelTraining.slider(
        'Energicnost', min_value=10, max_value=20, value=15)
    instrumentalness = modelTraining.slider('Instrumentalnost', min_value=10,
                                            max_value=20, value=15)
    key = modelTraining.slider('Kljuc', min_value=10, max_value=20, value=15)
    modelTraining.slider('Zivost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Glasnoca', min_value=10, max_value=20, value=15)
    modelTraining.slider('Audio mode', min_value=10, max_value=20, value=15)
    modelTraining.slider('Govornost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Tempo', min_value=10, max_value=20, value=15)
    modelTraining.slider('Vremenski potpis', min_value=10,
                         max_value=20, value=15)
    modelTraining.slider('Audio valencija', min_value=10,
                         max_value=20, value=15)

    num = st.number_input(
        "Higher precision step",
        min_value=1.0,
        max_value=5.0,
        step=1e-6,
        format="%.5f")

st.write(num)
st.write(key)
