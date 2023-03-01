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

def salt_and_papper_noise(img,value):
  # value must be from 0 to 255
  imp_noise=np.zeros((340,510),dtype=np.uint8)
  cv2.randu(imp_noise,0,255) #fill array with random distribution
  imp_noise=cv2.threshold(imp_noise,value,255,cv2.THRESH_BINARY)[1]
  in_img=cv2.add(img,imp_noise)
  plt.imshow(in_img,cmap='gray')
  plt.axis("off")                                                                    