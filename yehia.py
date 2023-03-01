import matplotlib.pyplot as plt

def Histogram(image):
    data = image.flatten() # converting the 2-d array to 1-d array
    plt.hist(data) # drawing the histogram for the image
    histogram_path = "histogram.png"
    plt.savefig(histogram_path) # saving the histogram
    return histogram_path