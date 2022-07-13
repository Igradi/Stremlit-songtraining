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
    st.write(song_data.describe())

with modelTraining:
    st.header('Time to train the model')
    st.text('Choose parameterers')

    duration = st.number_input(
        'Trajanje u ms', min_value=0.0, max_value=1800000.0, step=100.0)

    acousticness = st.number_input(
        'Akusticnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    danceability = st.number_input(
        'Plesnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    energy = st.number_input(
        'Energicnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    instrumentalness = st.number_input(
        'Instrumentalnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    key = st.slider('Kljuc', min_value=0, max_value=11, value=5)

    liveness = st.number_input(
        'Zivost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    loudness = st.number_input(
        'Glasnoca', min_value=-39.0, max_value=1.6, step=1e-2, format="%.3f")

    audio_mode = st.slider(
        'Audio mode', min_value=0, max_value=1, value=0)

    speechiness = st.number_input(
        'Govornost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    tempo = st.number_input('Tempo', min_value=0.0,
                            max_value=245.0, step=1e-2, format="%.3f")

    time_signature = st.slider(
        'Vremenski potpis', min_value=0, max_value=5, value=2)

    audio_valence = st.number_input(
        'Audio valencija', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")
