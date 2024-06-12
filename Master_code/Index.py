from Nonlinear_diff import *
from CDD import *
from efficiency import *
from PIL import Image
from skimage import color
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
from skimage.util import random_noise

# Load the images
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
tv_restored = io.imread(r"C:\Users\kizer\Master_code\inpainted_zoom.png")
gray_image = color.rgb2gray(original)

# Generate a mask
mask = generate_mask(gray_image)



#Apply NDF
#restored_NDF = nonlinearDiffusionFilter(rgb2gray(original))
# Convert the image to a format suitable for saving
#restored_NDF_uint8 = img_as_ubyte(restored_NDF)
# Save the image as PNG
#im = Image.fromarray(restored_NDF_uint8)
#im.save("restored_NDF.png")


# Example CDD:
# Define the g function based on curvature (e.g., g(s) = 1 / (1 + s) for s > 0)
g = lambda s: 1 / (1 + s)
# Inpaint the image (assuming `image` and `mask` are defined)
inpainted_image = cdd_inpainting(original, mask, g, iterations=1000, tau=0.1)


# Calculate PSNR
psnr_value = psnr(inpainted_image, gray_image)
print(f"PSNR: {psnr_value} dB")

if psnr_value < 20:
    print("the image has poorly been conserved")
elif 20 < psnr_value < 40: 
    print("The restored image is good")
else: 
    print("The quality of the restored image is excellent")