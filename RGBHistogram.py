import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave

def rgb_histogram(image):
    histogram_rgb_path = "images/workspace/histogram-rgb.png"
    img_r = image[:, :, 0]; img_g = image[:, :, 1]; img_b = image[:, :, 2] # separating the R, G, and B channels into different arrays
    # Converting the 2-d arrays to 1-d arrays
    img_r = img_r.flatten()
    img_g = img_g.flatten()
    img_b = img_b.flatten()
    fig, ax = plt.subplots(3,2) # creating figure
    # Ploting the histograms for the data
    ax[0, 0].hist(img_r)
    ax[1, 0].hist(img_g)
    ax[2, 0].hist(img_b)
    ax[0,0].set_title("Normal Histogram",fontsize = 10)
    fig.tight_layout(pad=3) # adding space between the subplots
    # Ploting the distribution curves for the data
    ax[0, 1].hist(img_r,cumulative=True)
    ax[1, 1].hist(img_g,cumulative=True)
    ax[2, 1].hist(img_b,cumulative=True)
    ax[0,1].set_title("Normal Distribution Curve",fontsize = 10)
    fig.savefig(histogram_rgb_path) # saving the figure
    return histogram_rgb_path

def equalized_image_rgb(image):
    equalized_rgb_img_path = "images/workspace/equalized-rgb-image.png"
    img_r = image[:, :, 0]; img_g = image[:, :, 1]; img_b = image[:, :, 2] # separating the R, G, and B channels into different arrays
    # Getting the histograms for each color channel
    his_r, be_r = np.histogram(img_r.flatten(), bins = 256) 
    his_g, be_g = np.histogram(img_g.flatten(), bins = 256) 
    his_b, be_b = np.histogram(img_b.flatten(), bins = 256) 

    # Getting the probability density function
    pdf_r = his_r.astype(float)/sum(his_r) 
    pdf_g = his_g.astype(float)/sum(his_g) 
    pdf_b = his_b.astype(float)/sum(his_b)

    # Getting the cumulative distribution function for each color channel
    cdf_r = np.cumsum(pdf_r) 
    cdf_g = np.cumsum(pdf_g)
    cdf_b = np.cumsum(pdf_b)

    # Using linear interpolation to get equalized cclor channels
    new_r = np.interp(img_r,be_r,np.hstack((np.zeros(1),cdf_r))) 
    new_g = np.interp(img_g,be_g,np.hstack((np.zeros(1),cdf_g)))
    new_b = np.interp(img_b,be_b,np.hstack((np.zeros(1),cdf_b)))

    image = image.astype(float)
    image[:, :, 0] = new_r; image[:, :, 1] = new_g; image[:, :, 2] = new_b # Changing each channel value in the original image
    # with its equalized value

    imsave(equalized_rgb_img_path, image) # saving the equalized image
    return equalized_rgb_img_path
