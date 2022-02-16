from PIL import Image

def crop(img, scale):
    area = (0, 0, 900*scale, 500*scale)
    cropped_img = img.crop(area)
    cropped_img.show()
    return cropped_img


im1 = crop(Image.open("dog1.jpg"), 1)
