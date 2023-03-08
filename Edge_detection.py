import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import Filters as filters
from skimage.io import imsave


def non_max_suppression(gradient_matrix: np.ndarray, theta_matrix: np.ndarray):
    """Check if the pixels on the same direction are more or less intense than the ones being processed. Used with canny detector
    Args:
        gradient_matrix (np.ndarray): magnitude matrix
        theta_matrix (np.ndarray): theta matrix
    Returns:
        _type_: filtered matrix of the same type
    """
    m, n = gradient_matrix.shape
    # Create a matrix initialized to 0 of the same size of the original gradient intensity matrix
    Z = np.zeros((m, n), dtype=np.int32)

    # Identify the edge direction based on the angle value from the angle matrix
    angle = theta_matrix * 180. / np.pi
    angle[angle < 0] += 180

    # Check if the pixel in the same direction has a higher intensity than the pixel that is currently processed
    for i in range(1, m-1):
        for j in range(1, n-1):
            try:
                # Max intensity = 255 for white pixels
                q = 255
                r = 255

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_matrix[i, j+1]
                    r = gradient_matrix[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_matrix[i+1, j-1]
                    r = gradient_matrix[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_matrix[i+1, j]
                    r = gradient_matrix[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_matrix[i-1, j-1]
                    r = gradient_matrix[i+1, j+1]

                # If there are no pixels in the edge direction having more intense values, then the value of the current pixel is kept
                if (gradient_matrix[i, j] >= q) and (gradient_matrix[i, j] >= r):
                    Z[i, j] = gradient_matrix[i, j]
                else:
                    # Intensity value of the current pixel is set to 0
                    Z[i, j] = 0

            except IndexError as e:
                print(f'IndexError: {e}')
                pass

    # Return the image processed with the non-max suppression algorithm
    return Z

def double_threshold(img_grayscale: np.ndarray, low_TH_ratio: float = 0.05, high_TH_ratio: float = 0.09):
    """To identify(filter) the strong, weak and non-relevant pixels 
    Args:
        img (np.ndarray): original image matrix (grayscale)
        low_TH_ratio (float, optional): value of the low threshold. Defaults to 0.05.
        high_TH_ratio (float, optional): value of the high threshold.. Defaults to 0.09.
    Returns:
        _type_: filtered matrix of the same type and shape of the original image
    """

    # Calculating both thresholds
    high_TH = img_grayscale.max() * high_TH_ratio
    low_TH = high_TH * low_TH_ratio

    M, N = img_grayscale.shape
    result_matrix = np.zeros((M, N), dtype=np.int32)

    weak_value = np.int32(25)
    strong_value = np.int32(255)

    # All pixels having intensity higher than high_TH are flagged as strong
    strong_i, strong_j = np.where(img_grayscale >= high_TH)

    # All pixels having intensity between both thresholds are flagged as weak
    weak_i, weak_j = np.where(
        (img_grayscale <= high_TH) & (img_grayscale >= low_TH))

    # All pixels having intensity lower than low_TH are flagged as non-relevant
    zeros_i, zeros_j = np.where(img_grayscale < low_TH)

    result_matrix[strong_i, strong_j] = strong_value
    result_matrix[weak_i, weak_j] = weak_value

    # result_matrix contains only 2 pixel intensity categories (strong and weak)
    return (result_matrix, weak_value, strong_value)
def hysteresis(img_grayscale: np.ndarray, weak_value: int, strong_value: int = 255):
    """Transform weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one
    Args:
        img (np.ndarray): original image (grayscale)
        weak_value (int): low threshold value
        strong_value (int, optional): high threshold value. Defaults to 255.
    Returns:
        np.ndarray: filtered matrix of the same type and shape of the original image
    """

    M, N = img_grayscale.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            # Check for each weak pixel
            if (img_grayscale[i, j] == weak_value):
                try:
                    # Check for the surrounding pixels (box)
                    if ((img_grayscale[i+1, j-1] == strong_value) or (img_grayscale[i+1, j] == strong_value) or (img_grayscale[i+1, j+1] == strong_value)
                        or (img_grayscale[i, j-1] == strong_value) or (img_grayscale[i, j+1] == strong_value)
                            or (img_grayscale[i-1, j-1] == strong_value) or (img_grayscale[i-1, j] == strong_value) or (img_grayscale[i-1, j+1] == strong_value)):
                        img_grayscale[i, j] = strong_value
                    else:
                        # Assign zero to weak pixels
                        img_grayscale[i, j] = 0
                except IndexError as e:
                    print(f'IndexError: {e}')
                    pass
    # return final form of the image using canny edges detector algorithm
    return img_grayscale
def sobel_kernels(image: np.ndarray):
    img_grayscale = np.copy(image)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = filters.apply_convolution(img_grayscale, Kx)
    Iy = filters.apply_convolution(img_grayscale, Ky)

    magnitude_matrix = np.hypot(Ix, Iy)
    magnitude_matrix = magnitude_matrix / magnitude_matrix.max() * 255
    theta_matrix = np.arctan2(Iy, Ix)

    return (magnitude_matrix, theta_matrix)

def canny_detector(image: np.ndarray,kernal_size,sigma):
    """Canny edge detector algorithm
    Args:
        image (np.ndarray): original image (grayscale)
    Returns:
        _type_: matrix of the image with canny mask applied 
    """
    path = "images/output/canny_detection.jpeg"

    img_grayscale = np.copy(image)

    # Noise reduction; by applying Gaussian blur to smooth it
    gaussian_mask = filters.gaussian_kernal(kernal_size, kernal_size, sigma)
    masked_matrix = filters.apply_convolution(img_grayscale, gaussian_mask)

    # Gradient Calculation; by using Sobel kernels
    gradient_matrix, theta_matrix = sobel_kernels(masked_matrix)

    # Thinner edges; by using Non-Maximum Suppression algorithm
    thinner_edges_matrix = non_max_suppression(gradient_matrix, theta_matrix)

    # Double threshold; to identify the pixels of the image according to pre-defined thresholds
    threshold_matrix, weak_value, strong_value = double_threshold(
        thinner_edges_matrix)

    # Hysteresis; to check for surrounding weak pixels if they have a strong value for better edges.
    final_matrix = hysteresis(threshold_matrix, weak_value, strong_value)
    plt.imshow(final_matrix, cmap="gray")
    plt.axis("off")
    plt.savefig(path)    
    return path

    # canny_detector(cv2.imread("images/original1.jpeg",0))


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # # Apply Equal Padding to All Sides
    # if padding != 0:
    #     imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
    #     imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    #     print(imagePadded)
    # else:
    #     imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * image[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def histogram(image):
    histogram_path = "images/output/histogram.png"
    dist_path = "images/output/dis.png"
    data = image.flatten() # converting the 2-d array to 1-d array
    # fig.tight_layout(pad=3) # adding space between the subplots
    bins = np.arange(256)
    plt.hist(data, bins=bins) # ploting the histogram for the data
    plt.title("Normal Histogram",fontsize = 10)
    plt.savefig(histogram_path)

    fig, ax = plt.subplots(1,2)
    # np.histogram(data)
    # ax[0].step(bins,data)
    ax[0].hist(data,cumulative=True,histtype="step") # ploting the distribution curve for the data
    ax[0].set_title("Normal Cumulative Curve",fontsize = 10)

    ax[1].hist(data, histtype="step")
    ax[1].set_title("Normal Distribution Curve", fontsize=10)
    fig.savefig(dist_path) # saving the figure

    return histogram_path, dist_path

def equalized_image(image):
    equalized_img_path = "images/output/equalized-image.png"
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
    plt.axis("off")
    plt.savefig(path)
    return path

def sobel_filter(image,mode):
    x_kernal = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) # vertical Sobel filter
    y_kernal = x_kernal.T # horizontal Sobel filter
    image_x = convolve2D(image,x_kernal) # applying the vertical filter to the image
    image_y = convolve2D(image,y_kernal) # applying the horizontal filter to the image
    return paths(mode,"images/output/sobel-x.png",image_x,"images/output/sobel-y.png",image_y,"images/output/new-sobel.png")

def roberts_filter(image,mode):
    kernal_x = np.array([[1,0],[0,-1]]) # vertical roberts filter
    kernal_y = np.flip(kernal_x,0).T #horizontal roberts filter
    image_x = convolve2D(image,kernal_x) # applying the 1st filter to the image
    image_y = convolve2D(image,kernal_y) # applying the 2nd filter to the image
    return paths(mode,"images/output/roberts-x.png",image_x,"images/output/roberts-y.png",image_y,"images/output/new-roberts.png")

def prewitt_filter(image,mode):
    kernal_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]]) # x-axis filter
    kernal_y = kernal_x.T # y-axis filter
    image_x = convolve2D(image,kernal_x) # applying the 1st filter to the image
    image_y = convolve2D(image,kernal_y) # applying the 2nd filter to the image
    return paths(mode,"images/output/prewitt-x.png",image_x,"images/output/prewitt-y.png",image_y,"images/output/new-prewitt.png")
