import streamlit as st
import Filters as filters
from PIL import Image
import Histograms as freq_filters
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2
import os
import NormEqua as NE 
import Edge_detection as edge_detection
import yehia as y
import RGBHistogram as rhis
import Thresholding as thresh

# vars
num_of_white_PX=0
num_of_black_PX=0
denoise_option=0
filters_option=0
filter_size=0
kernal_length=0

path='images'
st.set_page_config(page_title="Image Processing",layout="wide")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def modes(varaible):
    if varaible == "Vertical":
        mode = 0
    elif varaible is "Horizontal":
        mode = 1
    else:
        mode = 2
    return mode

def show_output(file_path):
    st.image(file_path)

col1, col2 = st.columns(2)
with st.sidebar:
    st.title('Upload an image')
    uploaded_file = st.file_uploader("", accept_multiple_files=False, type=['jpg','png','jpeg','webp'])
    st.title("Options")
    option = st.selectbox("",["Apply filter","Apply noise","Edge detection", "Calculate histogram","Hybrid image","Normalize","Equalize", "RGB Histogram", "Thresholding"])

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
        swap_flag = st.checkbox('Swap images')
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
            if swap_flag:
                temp=uploaded_file
                uploaded_file=sec_uploaded_file
                sec_uploaded_file=temp
            freq_filters.apply_hybrid_filter(uploaded_file,sec_uploaded_file,cutoff_lpf,cutoff_hpf)
            st.image("images/output/hybrid.png")
            


# Apply filter case 
elif option == "Apply filter":
    if uploaded_file is not None:
        with st.sidebar:
            # st.subheader("Filter type")
            Filter_type = st.radio("Filter type",["Frequency filters","Denoising filters"],horizontal=True)
            if Filter_type =='Frequency filters':
                # st.subheader("Frequency filter")
                filters_option = st.selectbox('Choose a filter',('Highpass Filter', 'Lowpass Filter'))
            elif Filter_type =='Denoising filters':
            # st.subheader("Denoise filter")
                denoise_option = st.selectbox('Choose a filter',('Gaussian', 'Median', 'Average'))
            # imagee = uploaded_file.read()
            # image_path2=os.path.join(path,sec_uploaded_file.name)

            image1=cv2.imread(image_path1,0)
            # image2=cv2.imread(image_path2,0)
            if filters_option == 'Highpass Filter':
                cut_off_freq = st.number_input('Cuttoff Frequency', min_value=1, value=10, step=5)
            elif filters_option == 'Lowpass Filter':
                cut_off_freq = st.number_input('Cuttoff Frequency', min_value=1, value=10, step=5)
            # denoising filters
            if denoise_option == 'Gaussian':
                with st.sidebar:
                    sigma = st.number_input('Sigma', min_value=1, value=20, step=2)
            # elif denoise_option == 'Medin':
            #         with st.sidebar:
            #             filter_size = st.number_input('filter size', min_value=0, value=10, step=5)
            elif denoise_option == 'Average' :      
                    with st.sidebar:
                        kernal_length = st.selectbox('Choose kernal length',('3x3', '5x5'))

                        # kernal= st.number_input('min_value=1, value=10, step=5)
                       

        input_img, resulted_img = st.columns(2)
        with input_img :
            st.title("Input image")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)
        with resulted_img:
            st.title("Output image")
            if filters_option == 'Highpass Filter':
                freq_filters.apply_highpass_filter(image1,cut_off_freq)
                st.image("images/output/highpass_filtered.jpeg")
            elif filters_option == 'Lowpass Filter':
                filtered_img_freq,lowPass_filter,img_fft=freq_filters.apply_lowPass_filter(image1,cut_off_freq)
                st.image("images/output/lowpass_filtered.jpeg")
            if denoise_option == 'Gaussian':
                filters.Apply_gaussian_filter(image_path1,sigma)
                st.image("images/output/Gaussian_filter.jpeg")
        
            elif denoise_option == 'Median':
                filters.apply_median_filter(image1)
                st.image("images/output/Median_filter.jpeg")

            elif denoise_option == 'Average':
                if kernal_length == '3x3':
                    filters.apply_average_filter(image1,3)
                if kernal_length == '5x5':
                    filters.apply_average_filter(image1,5)
                st.image("images/output/average_filter.jpeg")






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
        plt.axis('off')
        plt.imshow(NE.normalize(image1),cmap='gray')
        plt.savefig('images/output/normalized_photo.png')
        st.image('images/output/normalized_photo.png')
        
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
        NE.equalize(uploaded_file,'images/output/equalized_photo.png')
        st.image('images/output/equalized_photo.png')

# ----------------------------NOISE-------------------------------------------
elif option == "Apply noise":
    with st.sidebar:
        noise_option = st.selectbox('Choose a noise',('Gaussian', 'Uniform', 'Salt & pepper'))

    if noise_option == 'Gaussian':
            with st.sidebar:
                mean = st.number_input('Mean', min_value=20, value=100, step=5)
                sigma = st.number_input('Sigma', min_value=1, value=20, step=2)
    elif noise_option == 'Uniform':
            with st.sidebar:
                noise_value = st.number_input('Noise value', min_value=20, value=100, step=5)
    elif noise_option == 'Salt & pepper' :      
            with st.sidebar:
                num_of_white_PX = st.number_input('Num of white px', min_value=1, value=10, step=5)
                num_of_black_PX = st.number_input('Num of black px', min_value=1, value=10, step=5)
    input_img, resulted_img = st.columns(2)
    with input_img :
        st.title("Input image")
        image = Image.open(uploaded_file)
        st.image(uploaded_file)

    with resulted_img:
        st.title("Output image")
        image1=cv2.imread(image_path1,0)  
    if noise_option == 'Gaussian':
      
        filters.apply_Gaussian_Noise(image1,mean,sigma)
        st.image("images/output/Gaussian_noise.jpeg")
    
    elif noise_option == 'Uniform':
        filters.Apply_uniform_noise(image1,noise_value)
        st.image("images/output/Uniform_noise.jpeg")

    elif noise_option == 'Salt & pepper':
        filters.Apply_salt_and_papper_noise(image1,num_of_white_PX,num_of_black_PX)
        st.image("images/output/Salt & pepper_noise.jpeg")


    # ----------------------------Edge detection-------------------------------------------
elif option == "Edge detection":
    input_img, resulted_img = st.columns(2)

    with st.sidebar:
        edge_detect_options = st.selectbox('Choose detector',('Canny','Sobel','Prewitt','Roberts'))
    if edge_detect_options == 'Canny detector':
        with st.sidebar:
            canny_kernal = st.selectbox('Select kernal size',('3x3','5x5'))
            canny_sigma = st.number_input('Sigma', min_value=1, value=10, step=2)

        with input_img :
            st.title("Input image")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)

        with resulted_img:
            st.title("Output image")
            image1=cv2.imread(image_path1,0)  
            if canny_kernal == '3x3':
                edge_detection.canny_detector(image1, 3, canny_sigma)
                st.image("images/output/canny_detection.jpeg")
            elif canny_kernal == '5x5':
                edge_detection.canny_detector(image1, 5, canny_sigma)
                st.image("images/output/canny_detection.jpeg")
            # elif canny_kernal == '9x9':
            #     edge_detection.canny_detector(image1, 9, canny_sigma)
            #     st.image("images/output/canny_detection.jpeg")

    with st.sidebar:
        if edge_detect_options == "Sobel":
            sobel = st.selectbox("Sobel",("Vertical","Horizontal","Both"))
            show_output(y.sobel_filter(image,modes(sobel)))
        if edge_detect_options == "Prewitt":
            prewitt = st.selectbox("Prewitt",("Vertical","Horizontal","Both"))
            show_output(y.prewitt_filter(image,modes(prewitt)))
        if edge_detect_options == "Roberts":
            roberts = st.selectbox("Roberts",("Vertical","Horizontal","Both"))
            show_output(y.roberts_filter(image,modes(roberts)))

    # ----------------------------Histogram-------------------------------------------
elif option == "Calculate histogram":
    if uploaded_file is not None:
        image1=cv2.imread(image_path1, 0)
        with st.sidebar:
            modes = st.selectbox("Histogram",("Normal","Equalized"))

        input_img, resulted_img = st.columns(2)
        with input_img:
            st.title("Input image")
            image = Image.open(uploaded_file)
            st.image(image1)
        with resulted_img:
            if modes is "Normal":
                path_histogram = y.histogram(image1)
                st.image(path_histogram)
            if modes is "Equalized":
                path_equalized = y.equalized_image(image1)
                st.image(path_equalized) # showing the equalized image
                st.subheader("Equalized image histogram and distribution curve")
                image_equal = plt.imread(path_equalized)
                st.image(y.histogram(image_equal)) # applying histogram function to get histogram and distribution curve

#---------------------------------RGB Histogram-----------------------------------------
elif option == "RGB Histogram":
    if uploaded_file is not None:
        image1=cv2.imread(image_path1)
        with st.sidebar:
            modes = st.selectbox("RGB Histogram",("Normal", "Equalized"))

        input_img, resulted_img = st.columns(2)
        with input_img:
            st.title("Input image")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)
        with resulted_img:
            if modes == "Normal":
                if image1.ndim == 3:
                    path_histogram_rgb = rhis.rgb_histogram(image1)
                    st.image(path_histogram_rgb)
                elif image1.ndim == 2:
                    st.warning("Please upload a colored image")
            elif modes == "Equalized":
                if image1.ndim == 3:
                    path_equalized_rgb = rhis.equalized_image_rgb(image1)
                    st.image(path_equalized_rgb)
                    st.subheader("Equalized RGB image histogram and distribution curve")
                    rgb_image_equal = imread(path_equalized_rgb)
                    st.image(rhis.rgb_histogram(rgb_image_equal))
                elif image1.ndim == 2:
                    st.warning("Please upload a colored image")

#---------------------------------Thresholding------------------------------------------
elif option == "Thresholding":
    if uploaded_file is not None:
        threshold = 0; threshold_1 = 0; threshold_2 = 0; threshold_3 = 0; threshold_4 = 0
        thresh_mode = 0
        image1=cv2.imread(image_path1)
        with st.sidebar:
            edge_detect = st.selectbox("Thresholding Technique", ["Manual Thresholding", "Otsu's Thresholding"])
            if edge_detect == "Manual Thresholding":
                thresh_mode = 0
            elif edge_detect == "Otsu's Thresholding":
                thresh_mode = 1

            threshold_type = st.radio("Thresholding Type", ["Local", "Global"], horizontal=True)
            if threshold_type == "Local":
                if thresh_mode == 0:
                    threshold_1 = st.slider(label="Threshold Value 1", min_value=0, max_value=255, step=1)
                    threshold_2 = st.slider(label="Threshold Value 2", min_value=0, max_value=255, step=1)
                    threshold_3 = st.slider(label="Threshold Value 3", min_value=0, max_value=255, step=1)
                    threshold_4 = st.slider(label="Threshold Value 4", min_value=0, max_value=255, step=1)
                elif threshold_type == "Global":
                    if thresh_mode == 0:
                        threshold =st.slider(label="Threshold Value", min_value=0, max_value=255, step=1)
                
        input_img, resulted_img = st.columns(2)
        with input_img:
            st.title("Input image")
            image = Image.open(uploaded_file)
            st.image(uploaded_file)
        with resulted_img:
            if threshold_type == "Local":
                st.title("Output image")
                local_type = thresh.local_thresholding(image1, threshold_1, threshold_2, threshold_3, threshold_4, thresh_mode)
                st.image(local_type)
            elif threshold_type == "Global":
                st.title("Output image")
                global_type = thresh.global_thresholding(image1, threshold, thresh_mode)
                st.image(global_type)
    
        