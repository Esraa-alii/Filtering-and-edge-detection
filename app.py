import streamlit as st
import matplotlib.pyplot as plt
from skimage.io import imread
import yehia as y

def show_output(file_path):
    with col2:
        st.header("Output")
        st.image(file_path)

def modes(varaible):
    if varaible == "Vertical":
        mode = 0
    elif varaible is "Horizontal":
        mode = 1
    else:
        mode = 2
    return mode

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
            modes = st.radio("Histogram",["Normal","Equalized"],horizontal=True)
        if modes is "Normal":
            path_histogram = y.Histogram(image)
            show_output(path_histogram)
        # else:
            # mean = np.mean(data)
            # std = np.std(data)
            # plt.plot(norm.pdf(data,mean,std))
            # distribution_curve_path = "distribution_curve.png"
            # plt.savefig(distribution_curve_path)
            # show_output(distribution_curve_path)
    elif tab is "1":
        with st.sidebar:
            edge_detect = st.radio("Techniques",["Sobel","Roberts","Prewitt","Canny edge"],horizontal=True)
            if edge_detect is "Sobel":
                sobel = st.radio("Sobel",["Vertical","Horizontal","Both"],horizontal=True)
                show_output(y.sobel_filter(image,modes(sobel)))
            if edge_detect is "Roberts":
                roberts = st.radio("Roberts",["Vertical","Horizontal","Both"],horizontal=True)
                show_output(y.roberts_filter(image,modes(roberts)))
            if edge_detect is "Prewitt":
                prewitt = st.radio("Prewitt",["Vertical","Horizontal","Both"],horizontal=True)
                show_output(y.prewitt_filter(image,modes(prewitt)))
    else:
        st.write("Nothing")