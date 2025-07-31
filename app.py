import streamlit as st
import tensorflow
from PIL import Image
import numpy as np

st.write('Classifier')
classes = ['Cat','Dog']

@st.cache_resource

def load_model():
    model = tensorflow.keras.models.load_model(r"cat_dog_model.h5")
    return model

model = load_model()

def preprocess(image:Image.Image):
    img = image.resize((128,128))
    img = np.array(img)/255
    img = img.reshape(1,128,128,3)
    return img

upload_f = st.file_uploader('Choose an image', type = ['jpg', 'jpeg','png'])
if upload_f:
    st.write('File Upload')
    image = Image.open(upload_f)
    st.image(image, caption = 'Uploaded Images')
    preprocessed= preprocess(image)
    pred=model.predict(preprocessed)[0]
    class_index = np.argmax(pred)
    pred_class = classes[class_index]
    st.success(f'Predicted Class: {pred_class} ')
