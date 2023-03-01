import streamlit as st
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.stats import norm
import numpy as np
import yehia as freq

def show_output(file_path):
    with col2:
        st.header("Output")
        st.image(file_path)

col1, col2 = st.columns(2)
with st.sidebar:
    tab = st.radio("Tabs",["1","2","3"],horizontal=True)
    image_input = st.file_uploader("Input Image",type=["jpg","png","jpeg"])
if image_input is not None:
    image = imread(image_input,True) # reading image in garyscale
    rgb_image = imread(image_input) # reading the image in RGB to show it
    with col1:
        st.header("Input")
        st.image(rgb_image) # showing the RGB image
    if tab is "2":
        with st.sidebar:
            modes = st.radio("Required",["Histogram","Distribution Curve"],horizontal=True)
        if modes is "Histogram":
            path_histogram = freq.Histogram(image)
            show_output(path_histogram)
        # else:
            # mean = np.mean(data)
            # std = np.std(data)
            # plt.plot(norm.pdf(data,mean,std))
            # distribution_curve_path = "distribution_curve.png"
            # plt.savefig(distribution_curve_path)
            # show_output(distribution_curve_path)
    # elif tab is "1":
    #     with st.sidebar:
    #         edge_detect = st.radio("technique",["Sobel","Roberts","Prewitt","Canny"])
    #         if edge_detect is "Sobel":
    #             sobel = st.radio("Sobel ",["Vertical","Horizontal","Both"])
    #             if sobel is "Vertical":
    #                 fil.sobel_filter()
    else:
        st.write("Nothing")
