import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from helper import preprocess_data, modulation_output_map, plot_signal_plotly

def main():
    st.set_page_config(page_title="Deep-learning based automatic data modulation classification with impairments", layout="wide", initial_sidebar_state="expanded")

    st.title("Deep-learning based automatic data modulation classification with impairments")
    st.markdown("<h2 style='color: white;'>Group E018</h2>", unsafe_allow_html=True)
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload signal data", type="csv")

        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            data_dict = preprocess_data(uploaded_file)
            X = data_dict[next(iter(data_dict))]
            A = next(iter(data_dict))
            modulation_type = f'{A[0]}_{A[1]}dB_{A[2]}Hz'

    if uploaded_file is not None:
        left, middle, right = st.columns((2, 10, 2))
        with middle:
            plot_signal_plotly(X, modulation_type)

        model = load_model('cldnn_modelv11.keras')

        if st.button("Make Predictions", key="predict_button", help="Click to make predictions"):
            predictions = np.argmax(model.predict(X), axis=1)
            modulation_predicted = modulation_output_map[predictions[0]]
            # Display prediction result with custom styling
            st.markdown("<h2 style='text-align: center; color: green;'>Prediction Result</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; font-size: 30px;'>This signal uses <strong><span style='color: yellow'>{modulation_predicted}</span></strong> modulation</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
