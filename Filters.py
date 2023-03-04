import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def apply_Gaussian_Noise(img,mean,sigma):
  img_width,img_height=img.shape
  gauss_noise=np.zeros((img_width,img_height),dtype=np.uint8)
  cv2.randn(gauss_noise,mean,sigma)
  gauss_noise=(gauss_noise*0.5).astype(np.uint8)
  gauss_noise_img=cv2.add(img,gauss_noise)
  plt.imshow(gauss_noise_img,cmap='gray')
  plt.axis("off")


def Apply_uniform_noise(img,noise_value):
  # we create a uniform distribution whose lower and upper bounds are the minimum and maximum pixel values (0 and 255 respectively) along the dimensions of the image.
  img_width,img_height=img.shape
  uni_noise=np.zeros((img_width,img_height),dtype=np.uint8)
  cv2.randu(uni_noise,0,255)
  uni_noise=(uni_noise*noise_value).astype(np.uint8)
  un_img=cv2.add(img,uni_noise)
  plt.imshow(un_img,cmap='gray')
  plt.axis("off")

def Apply_salt_and_papper_noise(img,num_of_white_PX,num_of_black_PX):
     # Getting the dimensions of the image
    row , col = img.shape  
    # Randomly pick some pixels in the image for coloring them white
    number_of_pixels = random.randint(0, num_of_white_PX)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_axis=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_axis=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_axis][x_axis] = 255
          
    # Randomly pick some pixels in the image for coloring them black
    number_of_pixels = random.randint(0 , num_of_black_PX)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_axis=random.randint(0, row - 1)
        # Pick a random x coordinate
        x_axis=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_axis][x_axis] = 0
    plt.imshow(img,cmap='gray')
    plt.axis("off")
    
    # median filter
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final
  
# img = Image.open("Unoise girl image.jpeg")
# arr = np.array(img)
# removed_noise = median_filter(arr, 5) 
# img1 = Image.fromarray(removed_noise)

# plt.figure(figsize=(15,6))
# plt.subplot(121)
# plt.title("noised image")
# plt.imshow(img, vmin=0, vmax=255)
# plt.axis('off')
# plt.imshow(img, cmap="gray")
# plt.subplot(122)
# plt.title("Median Filtered")
# plt.imshow(img1, cmap="gray")
# plt.axis('off')

# average filter
def Apply_average_filter(img):
  img_width, img_height = img.shape

  # Develop Averaging filter(3, 3) kernal
  kernal = np.ones([3, 3], dtype = int)
  kernal = kernal / 9

  # Convolve the 3X3 mask over the image
  img_new = np.zeros([img_width, img_height])

  for i in range(1, img_width-1):
    for j in range(1, img_height-1):
      temp = img[i-1, j-1]*kernal[0, 0]+img[i-1, j]*kernal[0, 1]+img[i-1, j + 1]*kernal[0, 2]+img[i, j-1]*kernal[1, 0]+ img[i, j]*kernal[1, 1]+img[i, j + 1]*kernal[1, 2]+img[i + 1, j-1]*kernal[2, 0]+img[i + 1, j]*kernal[2, 1]+img[i + 1, j + 1]*kernal[2, 2]
      img_new[i, j]= temp
      
  img_new = img_new.astype(np.uint8)
  plt.imshow(img_new, cmap="gray")
  plt.axis('off')

def apply_convolution(img_grayscale, mask):
    """a function that performs the convolution of the gaussian filter + the input image 
    Args:
        img_grayscale (array): the input image 
        mask (array): the mask(kernel)
    Returns:
        filtered_img
    """
    row, col = img_grayscale.shape
    masked_row, masked_col = mask.shape
    new = np.zeros((row + masked_row - 1, col + masked_col - 1))
    # setting the boundries of the image array
    masked_col = masked_col//2
    masked_row = masked_row//2
    filtered_img = np.zeros(img_grayscale.shape)
    new[masked_row:new.shape[0]-masked_row, masked_col:new.shape[1]-masked_col] = img_grayscale
    # looping over the image row indices
    for i in range(masked_row, new.shape[0]-masked_row):

        # looping over the image coloumn indices
        for j in range(masked_col, new.shape[1]-masked_col):
            temp = new[i-masked_row:i+masked_row+1, j-masked_row:j+masked_row+1]
            result = temp*mask
            filtered_img[i-masked_row, j-masked_col] = result.sum()

    return filtered_img

def gaussian_kernal(width, height, sigma):
    """_summary_
    Args:
        width : rows
        height : columns
        sigma: the standard deviation 
    Returns:
        gaussian: the filter array
    """
    # empty array
    gaussian = np.zeros((width, height))
    # setting the boundries of the filter
    width = width//2
    height = height//2
    # looping over rows
    for x in range(-width, width + 1):
 
        # looping over rows
        for y in range(-height, height + 1):
            # applying the equation of gaussian
            x1 = sigma*(2*np.pi)**2
            x2 = np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+width, y+height] = (1/x1)*x2

    return gaussian

def Apply_gaussian_filter(image_path: str ,sigma):
    """ function that calls the gaussian_filter and correlation functions and does some conversions
    Args:
        image_path (str): input image
    Returns:
        gaussian_filtered_img:
    """
    # creating an og_image object
    og_image = Image.open(image_path)
    gray_image = ImageOps.grayscale(og_image)

    # Convert it to numpy array
    img_grayscale = np.array(gray_image)

    # kernal size 9x9
    kernal = gaussian_kernal(9, 9, sigma)
    after_convolution = apply_convolution(img_grayscale, kernal)

    gaussian_filtered_img = after_convolution.astype(np.uint8)

   
    plt.imshow(gaussian_filtered_img,cmap='gray')    
    plt.axis('off')               
    # return gaussian_filtered_img

# guassian filter

# def gkernel(l=3, sig=2):
   
#     # Gaussian Kernel Creator via given length and sigma
#     ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
#     xx, yy = np.meshgrid(ax, ax)
#     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
#     return kernel / np.sum(kernel)

# img = cv2.imread('/Users/rawanghanemhmx/Desktop/Filters/Unoise girl image.jpeg') # Reading Image
# g_kernel = gkernel(3,2) # Create gaussian kernel with 3x3(odd) size and sigma equals to 2
# print("Gaussian Filter: ",g_kernel) # show the kernel array
# dst = cv2.filter2D(img,-1,g_kernel) #convolve kernel with image
# plt.figure(figsize=(15,6))
# plt.subplot(121)
# plt.title("Noised image")
# plt.imshow(img, vmin=0, vmax=255)
# plt.axis('off')
# plt.imshow(img)
# plt.subplot(122)
# plt.title("gaussian Filtered")
# plt.imshow(img_new, vmin=0, vmax=255)
# plt.axis('off')
# plt.imshow(img_new, cmap="gray")
