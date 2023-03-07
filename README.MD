# Filtering and edge detection

A web application that use python script for image processing such as adding noise, denoising and detect the edges.

> ## Table of Contents

- [Built with](#Built-with)
- [Deployment](#Deployment)
- [Design](#Design)
- [Features](#Features)
- [Authors](#Authors)


> ## Built with

![programming language](https://img.shields.io/badge/programmig%20language-Python-red)
![Framework](https://img.shields.io/badge/Framework-Streamlit-blue)
![styling](https://img.shields.io/badge/Styling-CSS-ff69b4)


> ## Deployment

 Install streamlit

```bash
  pip install streamlit
```

To start deployment 
```bash
  streamlit run main.py
```

> ## 🖌️ Design

![main widow](./Demo/main.png)
<!-- * **side bar** -->

<!-- <img  src="./images/images/newSideBar.png"  alt="DSP - sidebar1"  height="400"  /> -->

> ## Features
###  Apply Filter 

#### A. Frequency filters

1. Apply Highpass filter

![main widow](./Demo/highpass.gif)

2. Apply Lowpass filter

![main widow](./Demo/lowpass.gif)

#### B. Denoising filters
1. Apply Gaussian filter

![main widow](./Demo/gaussFilter.gif)

2. Apply Median filter

![main widow](./Demo/medianFilter.gif)

3. Apply Average filter 

![main widow](/Demo/averageFilter.gif)

###  Apply Noise
1. Gaussian noise
![main widow](./Demo/gaussNoise.gif)

2. Uniform noise
![main widow](./Demo/uniform.gif)

3. Salt and pepper noise
![main widow](./Demo/s%26p.gif)

###  Edge detection
1. Canny detector
![main widow](./Demo/canny.gif)

2. Sobel detector

3. Prewitt detector

4. Roberts detector

###  Histogram
1. Normal Histogram 

2. Equalize Histogram

###  Equalize the image 
![main widow](./Demo/Equalize.gif)

###  Normalize the image 
![main widow](./Demo/normalize.gif)

###  Hybrid image
![main widow](./Demo/hybrid.gif)


> ## 🔗 Authors
- Esraa Ali         
sec : 1   BN : 12

- Rawan Abdulhamid  
sec : 1   BN : 33

- Mostafa Mahmoud   
sec : 2   BN : 37

- Omar Mustafa      
sec : 2   BN : 5  

- Yehia Said        
sec : 2   BN : 53 


All rights reserved © 2023 to Team 9 - Systems & Biomedical Engineering, Cairo University (Class 2024)