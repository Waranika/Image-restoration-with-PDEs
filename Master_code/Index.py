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
from skimage.filters import gaussian
import cv2

# Function to generate a mask with a small square in the middle
def generate_square_mask(image, square_size=50):
    mask = np.zeros(image.shape, dtype=bool)
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = square_size // 2
    mask[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = True
    return mask


def generate_gaussian_noise_mask(image, noise_level=0.1, threshold_factor=1.0):
    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level, image.shape)
    
    # Calculate threshold based on noise level and factor
    threshold = noise_level * threshold_factor
    
    # Create a boolean mask where the noise exceeds the threshold
    mask = np.abs(noise) > threshold
    
    return mask


# Function to generate a mask with random pixels set to 0
def generate_random_pixel_mask(image, mask_ratio=0.1):
    # Create a mask with the same shape as the image, initialized to False
    mask = np.zeros(image.shape, dtype=bool)
    # Calculate the number of pixels to be masked
    num_pixels = image.size
    num_masked_pixels = int(num_pixels * mask_ratio)
    # Randomly choose indices to set to 0
    indices = np.random.choice(num_pixels, num_masked_pixels, replace=False)
    # Set the chosen indices in the mask to True
    mask.flat[indices] = True
    return mask


# Load the images
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
gray_image = color.rgb2gray(original)
'''
noise_level = 0.1  # Adjust this value to change the noise intensity
threshold_factor = 1.5  # Adjust this value to control mask density
mask = generate_gaussian_noise_mask(gray_image, noise_level, threshold_factor)
  
masked_image = np.copy(gray_image)
masked_image[mask] = 0
'''

# Generate a mask with a small square in the middle
mask = generate_square_mask(gray_image, square_size=50)

# Apply the mask to the grayscale image (set masked areas to 0)
masked_image = np.copy(gray_image)
masked_image[mask] = 0

# Apply the mask (highlight masked areas in red for visibility)
gray_image_rgb = np.stack([gray_image]*3, axis=-1)
gray_image_rgb[mask] = [1, 0, 0]  # Red color for masked areas
print("mse = ", mse(gray_image, masked_image))
print(cv2.PSNR(gray_image, masked_image))

'''
NDF = nonlinearDiffusionFilter(masked_image, iterations=14, lamb=0.31, tau=0.125)
print("mse = ", mse(gray_image, NDF))

# Calculate PSNR
#psnr_value = psnr(gray_image, inpainted_image_uint8)
psnr_value = cv2.PSNR(gray_image, NDF)
print(f"PSNR: {psnr_value} dB")

# Display the images using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(gray_image_rgb)
axes[1].set_title('Masked Image')
axes[1].axis('off')

axes[2].imshow(NDF, cmap='gray')
axes[2].set_title('Inpainted Image')
axes[2].axis('off')

plt.show()
'''
'''
# Gaussian filter
gauss_img = gaussian(masked_image, sigma=1)

print("mse = ", mse(gray_image, gauss_img))

# Calculate PSNR
#psnr_value = psnr(gray_image, inpainted_image_uint8)
psnr_value = cv2.PSNR(gray_image, gauss_img)
print(f"PSNR: {psnr_value} dB")


# Display the images using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Original Grayscale Image')
axes[0].axis('off')

axes[1].imshow(gray_image_rgb)
axes[1].set_title('Masked Image')
axes[1].axis('off')

axes[2].imshow(gauss_img, cmap='gray')
axes[2].set_title('Inpainted Image')
axes[2].axis('off')

plt.show()
'''

'''
# Run inpainting
uk, N= TV(gray_image, 1, mask, 5000, 0.5)
print("mse = ", mse(gray_image, uk))

# Calculate PSNR
#psnr_value = psnr(gray_image, inpainted_image_uint8)
psnr_value = cv2.PSNR(gray_image, uk)
print(f"PSNR: {psnr_value} dB")

print("mse = ", mse(gray_image, uk))

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

axes[2].imshow(uk, cmap='gray')
axes[2].set_title('Inpainted Image')
axes[2].axis('off')

plt.show()
'''

'''
# Example CDD:
p = 1
# Define the g function based on curvature 
g = lambda s: s**p
#g = lambda s: 1 / (1 + s)
# Inpaint the image 
inpainted_image = cdd_inpainting(masked_image, mask, g, iterations=5000, tau=0.03)

# Normalize the inpainted image to the range [0, 1]
inpainted_image_normalized = (inpainted_image - np.min(inpainted_image)) / (np.max(inpainted_image) - np.min(inpainted_image))

# Convert the inpainted image back to uint8 format for display
inpainted_image_uint8 = img_as_ubyte(inpainted_image_normalized)

# Calculate PSNR
#psnr_value = psnr(gray_image, inpainted_image_uint8)
psnr_value = cv2.PSNR(gray_image, inpainted_image)
print(f"PSNR: {psnr_value} dB")

print("mse = ", mse(gray_image, inpainted_image))

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