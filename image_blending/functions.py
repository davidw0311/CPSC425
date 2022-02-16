import numpy as np
from scipy import signal 
from PIL import Image
import cv2

# Part 2
# 1

def boxfilter(n):
# returns a box filter of size n by n, throws error if dimension is not odd
    if (n%2 == 0):
        raise AssertionError("Dimension must be odd")
    else:
        return np.full((n,n), 1.0/n**2)

# 2
def gauss1d(sigma):
    length = 6*sigma

    # rounds length to the next odd integer

    if int(length)%2 == 1:
        if length-int(length) > 0:
            length = int(length) + 2
        else:
            length = int(length)
    else:
        length = int(length) + 1
    
    filter = np.arange(-(length-1)//2, (length-1)//2+1, 1)
    filter = abs(filter)
    # create gaussian filter from Gaussian function
    filter = np.exp(-filter**2/(2*np.full_like(filter, sigma, dtype=np.float32)**2))

    # normalize the filter by dividing by the sum 
    filter /= np.sum(filter)

    return filter

# 3
def gauss2d(sigma):
    # first get 1d filter from sigma
    filter1d = gauss1d(sigma)

    # convolve 1d filter with its transpose
    filter2d = signal.convolve2d(filter1d[np.newaxis], (filter1d[np.newaxis]).T)

    return filter2d

# 4
# a)
def convolve_2d_manual(array, filter):

    kernel_size = filter.shape[0]

    # flatten the filter into 1D array and reverse the order to allow for faster convolution
    filter_array = np.flip(filter.flatten())
    
    # array to returned as the result of convolution
    convolved_array = np.zeros_like(array)
    
    # create a new 2d array that is the orginal array with zero padded borders
    # define padding width on each side
    padding = (kernel_size-1)//2
    # create zeros array with padded dimensions
    padded_array = np.zeros((array.shape[0] + padding*2, array.shape[1]+padding*2))
    # center of padded array is the original array
    padded_array[padding:-padding, padding:-padding] = array

    # perfom correlation, this operation is the same as convolution since the gaussian filter is symmetric
    for i in range(padding,padded_array.shape[0]-padding):
        for j in range(padding,padded_array.shape[1]-padding):

            # isolate section of image to be convolved, flatten and perfom element wise multiplication with 
            # the flipped filter array, and sum the resulting array to obtain convolved value
            section = padded_array[i-padding:i+padding+1,j-padding:j+padding+1]
            section_array = section.flatten()
            v = np.sum(section_array*filter_array)
            convolved_array[i-padding][j-padding] = v

    return convolved_array

# b)
def gaussconvolve2d_manual(array, sigma):
    filter = gauss2d(sigma)
    return convolve_2d_manual(array, filter)

# c)


# 5
# a
def gaussconvolve2d_scipy(array, sigma):
    filter = gauss2d(sigma)
    return signal.convolve2d(array, filter ,'same')

# Part 3
# 1
# applies blurring (low pass filter) to an image
def get_low_pass(filename, sigma):
    image = Image.open(filename)
    im_array = np.array(image, dtype=np.float32)
    r,g,b = im_array[:,:,0], im_array[:,:,1], im_array[:,:,2]

    blurred_r = gaussconvolve2d_manual(r, sigma)
    blurred_g = gaussconvolve2d_manual(g, sigma)
    blurred_b = gaussconvolve2d_manual(b, sigma)
    
    blurred_array = np.stack((blurred_r, blurred_g, blurred_b), -1)
    return blurred_array

# applies a high pass filter to an image
def get_high_pass(filename, sigma):
    image = Image.open(filename)
    im_array = np.array(image, dtype=np.float32)
    blurred_array = get_low_pass(filename, sigma)
    
    return im_array-blurred_array



if __name__ == "__main__":
    print("helper functions for Assignment1")