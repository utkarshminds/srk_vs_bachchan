import streamlit as st
import keras 
import tensorflow
import os
import numpy 
from PIL import Image 

@st.cache_data
def load_best_model():
    bestmodel = keras.load_model('bestmodel.keras')
    return bestmodel


st.title('SRK vs Bachchan classifier')

image_file = st.file_uploader('Upload the image', type=['jpg'])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption='uploaded image', use_column_width=True)

    if st.button('Predict'):
        prediction = make_prediction(image)
        st.write(prediction)
