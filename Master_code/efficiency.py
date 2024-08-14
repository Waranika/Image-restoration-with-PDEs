import numpy as np

def mse(imageA, imageB):
    # sum of the squared difference between the two images;
    # the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err


#CHANGE TO JUST THE MASK 
def psnr(original, restored):
    mse_value = mse(original, restored)
    if mse_value == 0:
        # Means no error
        return 100
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value

from skimage import io
