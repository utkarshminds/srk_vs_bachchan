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

def make_prediction(image):
    img_resized = image.resize((200,200), Image.LANCZOS)
    image_array = numpy.expand_dims(img_resized, axis=0)
    bestmodel = load_best_model()
    preds = bestmodel.predict(image_array)

    class_final = (preds > 0.5).astype("int32")

    if class_final == 0:
        return 'bachchan'
    else:
        return 'srk'


st.title('SRK vs Bachchan classifier')

image_file = st.file_uploader('Upload the image', type=['jpg'])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption='uploaded image', use_column_width=True)

    if st.button('Predict'):
        prediction = make_prediction(image)
        st.write(prediction)
