from Nonlinear_diff import *
from CDD import *
from efficiency import *
from PIL import Image
from skimage import color
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
from skimage.util import random_noise

# Function to generate a mask with random noise
def generate_mask(image, noise_amount=0.2):
    noisy_image = random_noise(image, mode='s&p', amount=noise_amount)
    mask = noisy_image != image
    return mask

# Load the images
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
tv_restored = io.imread(r"C:\Users\kizer\Master_code\inpainted_zoom.png")
gray_image = color.rgb2gray(original)

# Save the gray image as PNG
gray_image_uint8 = img_as_ubyte(gray_image)
im = Image.fromarray(gray_image_uint8)
im.save("gray.png")


# Generate a mask
mask = generate_mask(gray_image)

# Create a color version of the grayscale image to overlay the mask
gray_image_rgb = np.stack([gray_image_uint8]*3, axis=-1)

# Apply the mask (highlight masked areas in red for visibility)
#gray_image_rgb[mask] = [0, 0, 0]  # Red color for masked areas

# Save the gray image with the mask overlay as PNG
#im_with_mask = Image.fromarray(gray_image_rgb)
#im_with_mask.save("gray_with_mask.png")

# Apply NDF
#restored_NDF = nonlinearDiffusionFilter(rgb2gray(original))
#Convert the image to a format suitable for saving
#restored_NDF_uint8 = img_as_ubyte(restored_NDF)
#Save the image as PNG
#im = Image.fromarray(restored_NDF_uint8)
#im.save("restored_NDF.png")

# Example CDD:
# Define the g function based on curvature (e.g., g(s) = 1 / (1 + s) for s > 0)
g = lambda s: 1 / (1 + s)
# Inpaint the image (assuming `image` and `mask` are defined)
inpainted_image = cdd_inpainting(gray_image, mask, g, iterations = 1000,  tau=0.01)
# Normalize the inpainted image to the range [0, 1]
inpainted_image_normalized = (inpainted_image - np.min(inpainted_image)) / (np.max(inpainted_image) - np.min(inpainted_image))
# Convert the inpainted image back to uint8 format for saving and display
inpainted_image_uint8 = img_as_ubyte(inpainted_image_normalized)
# Save the inpainted image
im = Image.fromarray(inpainted_image_uint8)
im.save("inpainted_image_fixed.png")

# Calculate PSNR
psnr_value = psnr(gray_image, inpainted_image_uint8)
print(f"PSNR: {psnr_value} dB")

if psnr_value < 20:
    print("The image has poorly been conserved")
elif 20 < psnr_value < 40: 
    print("The restored image is good")
else: 
    print("The quality of the restored image is excellent")
