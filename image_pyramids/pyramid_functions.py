from http.client import TEMPORARY_REDIRECT
from random import gauss
from re import I
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math
from scipy import signal
import ncc
import scipy as sp

# takes in an image with data type as numpy array
# returns a list for the gaussian pyramid, with each image as numpy array
def MakeGaussianPyramid(image, scale, minsize):
    
    sigma = 1/(2*scale)
    gaussian_pyramid = []
    gaussian_pyramid.append(image)

    # continue loop while the smallest dimension of the image is > minsize
    while min(image.shape[0], image.shape[1]) > minsize:
        
        # if the image is multi-channel, separately filter each channel and recombine
        if len(image.shape) == 3:
            channels = [image[:,:,c] for c in range(image.shape[-1])]
            channels = [sp.ndimage.gaussian_filter(c, sigma) for c in channels]
            image = np.dstack(channels)
        # if the image is single channel 
        else:
            image = sp.ndimage.gaussian_filter(image, sigma)
        
        # resize image after blurring and append to gaussian pyramid
        width, height = int(image.shape[1]*scale), int(image.shape[0]*scale)
        resized = Image.fromarray(np.uint8(image)).resize((width, height), Image.BICUBIC)
        image = np.array(resized, dtype=np.float32)
        gaussian_pyramid.append(image)
       
    return gaussian_pyramid

# list of numpy arrays as input for the pyramid
def ShowGaussianPyramid(pyramid):
    height, width = pyramid[0].shape[0], pyramid[0].shape[1]

    # determine the scale for each successive image
    scale = pyramid[1].shape[0]/pyramid[0].shape[0]

    # initialize width of frame using geometric series from the scale
    image = Image.new("RGB", (int(width * (1/(1-scale))), height), color=(255,255,255))
    
    offset_x, offset_y = 0,0
    for i, im in enumerate(pyramid):
        image.paste(Image.fromarray(np.uint8(im)),(offset_x, offset_y))
        offset_x += im.shape[1]
    return image


# pyramid: list of images of the pyramid, as numpy arrays
# template: PIL image of the template
# threshold: int
def FindTemplate(pyramid, template, threshold):
    
    base_image = Image.fromarray(pyramid[0]).convert('RGB')

    TEMPLATE_WIDTH = 15
    TEMPLATE_HEIGHT = int(template.height / (template.width/TEMPLATE_WIDTH))
    template = template.resize((TEMPLATE_WIDTH,TEMPLATE_HEIGHT), Image.BICUBIC)

    # function to draw bounding box around coordinates
    def draw_bbox(base_image, matching_coordinates, template, scale):

        for coordinate in matching_coordinates:

            # define template rectangle from correlation center on the current image scale
            x1 = int(coordinate[0]-template.width/2)
            y1 = int(coordinate[1]-template.height/2)
            x2 = int(coordinate[0]+template.width/2)
            y2 = int(coordinate[1]+template.height/2)

            # scale up coordinates to base image
            x1p = int(x1/scale)
            x2p = int(x2/scale)
            y1p = int(y1/scale)
            y2p = int(y2/scale)
     
            # draw the lines for the rectangle using resized coordinates
            draw = ImageDraw.Draw(base_image)
            draw.line((x1p,y1p,x2p,y1p), fill="red", width=2)
            draw.line((x1p,y1p,x1p,y2p), fill="red", width=2)
            draw.line((x1p,y2p,x2p,y2p), fill="red", width=2)
            draw.line((x2p,y1p,x2p,y2p), fill="red", width=2)
            del draw
        
        return base_image

    # iterate each layer of the image and perform ncc
    for i, layer_image in enumerate(pyramid[1:]):
        image = Image.fromarray(np.uint8(layer_image))
        coefficients = ncc.normxcorr2D(image, template)

        # find matching coordinates and convert to list (x,y) for all matching coordinates
        matching_indices = np.where(coefficients > threshold)
        matching_coordinates = np.array( [ [b,a] for a,b in zip(matching_indices[0], matching_indices[1]) ])

        base_image = draw_bbox(base_image, matching_coordinates, template, scale=(0.75)**(i+1))
        
    return base_image


def MakeLaplacianPyramid(image, scale, minsize):
    # start from a gaussian pyramid
    gp = MakeGaussianPyramid(image, scale, minsize)
    
    laplacian_pyramid = []

    for i in range(len(gp)-1):
        # obtain the higher resolution image from gaussian pyramid
        layer_image = gp[i]
        width = int(layer_image.shape[1])
        height = int(layer_image.shape[0])

        # obtain next lower resolution image in gaussian pyramid
        next_image = gp[i+1]
        
        # upsample and subtract from current layer
        upsampled_image = np.array(Image.fromarray(np.uint8(next_image)).resize((width, height), Image.BICUBIC), dtype=np.float32)
        laplacian_pyramid.append(layer_image-upsampled_image)

    # append the lowest resolution image (same for gaussain and laplacian pyramids)
    laplacian_pyramid.append(gp[-1])

    return laplacian_pyramid


def ShowLaplacianPyramid(pyramid):
    height, width = pyramid[0].shape[0], pyramid[0].shape[1]

    # determine the scale for each successive image
    scale = pyramid[1].shape[0]/pyramid[0].shape[0]

    # initialize width of frame using geometric series from the scale
    image = Image.new("RGB", (int(width * (1/(1-scale))), height), color=(255,255,255))
    
    # paste all images in pyramid in one horizontal strip
    offset_x, offset_y = 0,0
    for i in range(len(pyramid)):
        im = pyramid[i].copy()
        if i != len(pyramid)-1:
            im = im*2
            im += 128*2
        image.paste(Image.fromarray(np.uint8(im)),(offset_x, offset_y))
        offset_x += im.shape[1]
    
    return image


def ReconstructGaussianFromLaplacian(lPyramid):
    gaussian_pyramid = []
    
    image = lPyramid[-1]
    # top of the pyramids are the same
    gaussian_pyramid.insert(0, image)
    
    # start from the end of the pyramid (lowest res)
    i = len(lPyramid) - 1
    
    while i > 0:
        # get next higher res laplacian layer
        laplacian_layer = lPyramid[i-1]
        width, height = laplacian_layer.shape[1], laplacian_layer.shape[0]

        # upsample current image
        upsample = np.array(Image.fromarray(np.uint8(image)).resize((width, height), Image.BICUBIC), dtype=np.float32)
        
        # add with laplacian layer to get gaussian layer
        gaussian_layer = np.clip(upsample + laplacian_layer, 0, 255.0)
        gaussian_pyramid.insert(0, gaussian_layer) 

        # current gaussian layer becomes new starting point for next iteration
        image = gaussian_layer
        i -= 1
    return gaussian_pyramid


if __name__ == '__main__':
    print("")
