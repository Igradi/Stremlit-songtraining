from requests import head
import streamlit as st
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()


st.cache


def get_data(filename):

    song_data = pd.read_csv(filename)

    return song_data


with header:
    st.title('Dobrodosli u projekt')

song_data = get_data('data/song_data.csv')

with modelTraining:
    st.header('Time to train the model')
    st.text('Choose parameterers')
    modelTraining.slider('Popularnost pjesme', min_value=0,
                         max_value=100, value=50)
    modelTraining.slider('Trajanje u ms', min_value=10, max_value=20, value=15)
    modelTraining.slider('Akusticnost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Plesnost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Energicnost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Instrumentalnost', min_value=10,
                         max_value=20, value=15)
    modelTraining.slider('Kljuc', min_value=10, max_value=20, value=15)
    modelTraining.slider('Zivost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Glasnoca', min_value=10, max_value=20, value=15)
    modelTraining.slider('Audio mode', min_value=10, max_value=20, value=15)
    modelTraining.slider('Govornost', min_value=10, max_value=20, value=15)
    modelTraining.slider('Tempo', min_value=10, max_value=20, value=15)
    modelTraining.slider('Vremenski potpis', min_value=10,
                         max_value=20, value=15)
    modelTraining.slider('Audio valencija', min_value=10,
                         max_value=20, value=15)
