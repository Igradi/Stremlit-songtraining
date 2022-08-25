import plotly.express as px
import streamlit.components.v1 as components
import pandas as pd
from requests import head
import streamlit as st
import sklearn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
st.set_page_config(page_title="Song Recommendation", layout="wide")
model = pickle.load(open('model.sav', 'rb'))

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
output = st.container()


@st.cache
def get_data(filename):

    song_data = pd.read_csv(filename)

    return song_data


with header:
    st.title('Dobrodošli u projekt')
    st.image('songs.jpg')

with dataset:
    song_data = get_data('data/song_data.csv')
    st.set_option('deprecation.showPyplotGlobalUse', False)

with modelTraining:
    st.header(
        'Izaberite sve parametre kako biste predvidili popularnost svoje pjesme')
    st.text('Odaberite parametre pomoću slidera i formi')


def user_report():
    duration = st.number_input(
        'Trajanje u ms', min_value=0.0, max_value=1800000.0, step=100.0)

    acousticness = st.number_input(
        'Akustičnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    danceability = st.number_input(
        'Plesnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    energy = st.number_input(
        'Energičnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    instrumentalness = st.number_input(
        'Instrumentalnost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    key = st.slider('Kljuc', min_value=0, max_value=11, value=5)

    liveness = st.number_input(
        'Živost', min_value=0.0, max_value=1.0, step=1e-5, format="%.5f")

    loudness = st.number_input(
        'Glasnoća', min_value=-39.0, max_value=1.6, step=1e-2, format="%.3f")

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

    user_report_data = {
        'duration': duration,
        'acousticness': acousticness,
        'danceability': danceability,
        'energy': energy,
        'instrumentalness': instrumentalness,
        'key': key,
        'liveness': liveness,
        'loudness': loudness,
        'audio_mode': audio_mode,
        'speechiness': speechiness,
        'tempo': tempo,
        'time_signature': time_signature,
        'audio_valence': audio_valence,
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


with output:
    user_data = user_report()
    st.write('Parametri ste odabrali su: ')
    st.write(user_data)

    popularity = model.predict(user_data)
    st.subheader('Predviđena popularnost Vaše pjesme je: '+str(
        np.round(popularity[0], 2)))
