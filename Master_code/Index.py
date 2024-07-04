from Nonlinear_diff import *
from CDD import *
from efficiency import *
from PIL import Image
from skimage import color
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
from TV import *

# Function to generate a mask with a small square in the middle
def generate_square_mask(image, square_size=50):
    mask = np.zeros(image.shape, dtype=bool)
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = square_size // 2
    mask[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = True
    return mask

# Load the images
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
gray_image = color.rgb2gray(original)


# Generate a mask with a small square in the middle
mask = generate_square_mask(gray_image, square_size=50)

# Apply the mask to the grayscale image (set masked areas to 0 for inpainting)
masked_image = np.copy(gray_image)
masked_image[mask] = 0

# Apply the mask (highlight masked areas in red for visibility)
gray_image_rgb = np.stack([gray_image]*3, axis=-1)
gray_image_rgb[mask] = [1, 0, 0]  # Red color for masked areas

# Parameters
lamda, gama = 0.1, 0.1
iter = 1000

# Run inpainting
uk, rmse, num_iterations = tv1inpaint(masked_image, lamda, gama, iter, 214, 320)
print(f'RMSE: {rmse[-1]}, Iterations: {num_iterations}')

# Display the images using matplotlib

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(gray_image_rgb)
axes[1].set_title('Masked Image')
axes[1].axis('off')

axes[2].imshow(uk, cmap='gray')
axes[2].set_title('Inpainted Image')
axes[2].axis('off')

plt.show()
'''
# Example CDD:
p = 5
# Define the g function based on curvature 
g = lambda s: s**p
#g = lambda s: 1 / (1 + s)
# Inpaint the image 
inpainted_image = cdd_inpainting(masked_image, mask, g, iterations=50000, tau=0.001)

# Normalize the inpainted image to the range [0, 1]
inpainted_image_normalized = (inpainted_image - np.min(inpainted_image)) / (np.max(inpainted_image) - np.min(inpainted_image))

# Convert the inpainted image back to uint8 format for display
inpainted_image_uint8 = img_as_ubyte(inpainted_image_normalized)

# Calculate PSNR
psnr_value = psnr(gray_image, inpainted_image_uint8)
print(f"PSNR: {psnr_value} dB")

if psnr_value < 20:
    print("The image has poorly been conserved")
elif 20 < psnr_value < 40: 
    print("The restored image is good")
else: 
    print("The quality of the restored image is excellent")

# Display the images using matplotlib

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(gray_image_rgb)
axes[1].set_title('Masked Image')
axes[1].axis('off')

axes[2].imshow(inpainted_image_uint8, cmap='gray')
axes[2].set_title('Inpainted Image')
axes[2].axis('off')

plt.show()

'''