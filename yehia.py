import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from skimage.io import imsave


def histogram(image):
    histogram_path = "histogram.png"
    data = image.flatten() # converting the 2-d array to 1-d array
    fig, ax = plt.subplots(1,2) # creating figure
    fig.tight_layout(pad=3) # adding space between the subplots
    ax[0].hist(data) # ploting the histogram for the data
    ax[0].set_title("Normal Histogram",fontsize = 10)
    ax[1].hist(data,cumulative=True) # ploting the distribution curve for the data
    ax[1].set_title("Normal Distribution Curve",fontsize = 10)
    fig.savefig(histogram_path) # saving the figure
    return histogram_path

def equalized_image(image):
    equalized_img_path = "equalized-image.png"
    his, be = np.histogram(image.flatten()) # getting the histogram for the image
    pdf = his.astype(float)/sum(his) # getting the probability density function
    cdf = np.cumsum(pdf) # getting the cumulative distribution function for the image
    new_image = np.interp(image,be,np.hstack((np.zeros(1),cdf))) # using linear interpolation to get an equalized image
    imsave(equalized_img_path,new_image) # saving the equalized image
    return equalized_img_path

def paths(mode,path1,image1,path2,image2,path3):
    if mode == 0:
        path = path1
        plt.imshow(image1,cmap="gray")
    elif mode == 1:
        path = path2
        plt.imshow(image2,cmap="gray")
    else:
        image3 = np.sqrt(image1**2,image2**2) # Combining the 1st and 2nd image (corresponding to x and y respectively) in one image 
        path = path3
        plt.imshow(image3,cmap="gray")
    plt.savefig(path)
    return path

def sobel_filter(image,mode):
    x_kernal = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) # vertical Sobel filter
    y_kernal = x_kernal.T # horizontal Sobel filter
    image_x = convolve2d(image,x_kernal) # applying the vertical filter to the image
    image_y = convolve2d(image,y_kernal) # applying the horizontal filter to the image
    return paths(mode,"sobel-x.png",image_x,"sobel-y.png",image_y,"new-sobel.png")

def roberts_filter(image,mode):
    kernal_x = np.array([[1,0],[0,-1]]) # vertical roberts filter
    kernal_y = np.flip(kernal_x,0).T #horizontal roberts filter
    image_x = convolve2d(image,kernal_x) # applying the 1st filter to the image
    image_y = convolve2d(image,kernal_y) # applying the 2nd filter to the image
    return paths(mode,"roberts-x.png",image_x,"roberts-y.png",image_y,"new-roberts.png")

def prewitt_filter(image,mode):
    kernal_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]) # x-axis filter
    kernal_y = kernal_x.T # y-axis filter
    image_x = convolve2d(image,kernal_x) # applying the 1st filter to the image
    image_y = convolve2d(image,kernal_y) # applying the 2nd filter to the image
    return paths(mode,"prewitt-x.png",image_x,"prewitt-y.png",image_y,"new-prewitt.png")

# def canny_filter(image,mode):
#     return