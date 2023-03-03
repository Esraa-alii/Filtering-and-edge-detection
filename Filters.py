import numpy as np
import matplotlib.pyplot as plt
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