import streamlit as st
import Filters as filters
from PIL import Image
import Histograms as freq_filters
import matplotlib.pyplot as plt
import cv2
import os
import NormEqua as NE 

path='images'
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_output(file_path):
    st.image(file_path)

col1, col2 = st.columns(2)
with st.sidebar:
    st.title('Upload an image')
    uploaded_file = st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg'])
    
    option = st.radio("Options",["Apply filter","Calculate histogram","Hybrid image","Normalize","Equalize"],horizontal=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    plt.imread(uploaded_file)
    image_path1=os.path.join(path,uploaded_file.name)
    # print(image_path)
    # plt.savefig('i1.jpeg')
    # st.image(uploaded_file)



# hybrid image case 
if option == "Hybrid image":
    with st.sidebar:
        st.title('Upload the second image')
        sec_uploaded_file = st.file_uploader("", accept_multiple_files=False, type=['png','jpeg'])
        cutoff_lpf = st.number_input('Cuttoff lpf', min_value=1, value=10, step=5)
        cutoff_hpf = st.number_input('Cuttoff hpf', min_value=1, value=10, step=5)
        # image_path2=os.path.join(path,sec_uploaded_file.name)
    input_img, resulted_img = st.columns(2)
    with input_img :
        st.title("Input images")
        image = Image.open(uploaded_file)
        st.image(uploaded_file)
        if sec_uploaded_file is not None:
            image = Image.open(sec_uploaded_file)
            st.image(sec_uploaded_file)

    with resulted_img:
        st.title("Hybrid image")
        uploaded_file=cv2.imread(image_path1,0)
        if sec_uploaded_file is not None:
            image_path2=os.path.join(path,sec_uploaded_file.name)
            sec_uploaded_file=cv2.imread(image_path2,0)
            freq_filters.apply_hybrid_filter(uploaded_file,sec_uploaded_file,cutoff_lpf,cutoff_hpf)
            st.image("hybrid.png")


# Apply filter case 
elif option == "Apply filter":
    if uploaded_file is not None:
        with st.sidebar:
            st.subheader("Frequency filter")
            filters_option = st.selectbox('Choose a filter',('Highpass Filter', 'Lowpass Filter'))
            # imagee = uploaded_file.read()
            # image_path2=os.path.join(path,sec_uploaded_file.name)

            image1=cv2.imread(image_path1,0)
            # image2=cv2.imread(image_path2,0)
            if filters_option == 'Highpass Filter':
                cut_off_freq = st.number_input('Cuttoff Frequency', min_value=1, value=10, step=5)
            elif filters_option == 'Lowpass Filter':
                cut_off_freq = st.number_input('Cuttoff Frequency', min_value=1, value=10, step=5)

        input_img, resulted_img = st.columns(2)
        with input_img :
            st.title("Input image")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)
        with resulted_img:
            st.title("Output image")
            if filters_option == 'Highpass Filter':
                freq_filters.apply_highpass_filter(image1,cut_off_freq)
                st.image("highpass_filtered.jpeg")
            elif filters_option == 'Lowpass Filter':
                filtered_img_freq,lowPass_filter,img_fft=freq_filters.apply_lowPass_filter(image1,cut_off_freq)
                st.image("lowpass_filtered.jpeg")

# ---------------------------- NORMALIZATION ---------------------------------
elif option == "Normalize":
    image1=cv2.imread(image_path1,0)
    input_img, resulted_img = st.columns(2)
    with input_img :
        st.title("Input image")
        image = Image.open(uploaded_file)
        st.image(uploaded_file)
    with resulted_img:
        st.title("Output image")
        plt.imshow(NE.normalize(image1),cmap='gray')
        plt.savefig('normalized_photo.png')
        st.image('normalized_photo.png')
        
# ----------------------------- EQUALIZATION -----------------------------------
elif option == "Equalize" :
    image1=cv2.imread(image_path1,0)
    input_img, resulted_img = st.columns(2)
    with input_img :
        st.title("Input image")
        image = Image.open(uploaded_file)
        st.image(uploaded_file)
    with resulted_img:
        st.title("Output image")
        NE.equalize(uploaded_file,'equalized_photo.png')
        st.image('equalized_photo.png')