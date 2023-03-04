import streamlit as st
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
    with col1:
        st.header("Input")
        st.image(image) # showing the grayscale image
    if tab is "2":
        with st.sidebar:
            modes = st.radio("Histogram",["Normal","Equalized"],horizontal=True)
        if modes is "Normal":
            path_histogram = y.histogram(image)
            show_output(path_histogram)
        if modes is "Equalized":
            path_equalized = y.equalized_image(image)
            show_output(path_equalized) # showing the equalized image
            st.subheader("Equalized image histogram and distribution curve")
            image_equal = imread(path_equalized)
            st.image(y.histogram(image_equal)) # applying histogram function to get histogram and distribution curve
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