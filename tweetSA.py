import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import MyProcessingModule as processing


st.title('Twitter sentiment analysis')
st.write("""
This application is to predict sentiment of tweets.
"""
)

model_name = 'TwitterSentimentModel'

def load_model(model):
    return keras.models.load_model(model)

# Create a text element and let the reader know the data is loading.
model_load_state = st.text('Loading Model...')
# Load 10,000 rows of data into the dataframe.
model = load_model(model_name)
# Notify the reader that the data was successfully loaded.
model_load_state.text("Model already deployed")

tweet_input = st.text_input('Leave tweet content here')
entity_input = st.selectbox("Choose entity to increase prediction accuracy by category", processing.unique_entity(),index=0)
if st.button('Predict'):
    result = model.predict(processing.preprocessing(tweet_input,entity=entity_input))
    labelmodel = processing.getlabel()
    result_label = pd.DataFrame(result)
    result_label.columns = labelmodel.inverse_transform(result_label.columns)
    st.write("Tweet: ",tweet_input)
    st.write("Entity: ",entity_input)
    st.write(result_label)
    colorPalette=["#61412B","#DCC57D","#D57838","#FFCE33"]
    fig,ax= plt.subplots(1,1,figsize=(10,10))
    ax.pie(result_label.iloc[0],labels=result_label.columns,explode=[0.01,0.01,0.01,0.01],colors=colorPalette,autopct='%1.1f%%')
    st.write(fig)