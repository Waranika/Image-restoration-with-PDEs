from TV import *
from efficiency import *
import numpy as np
import cv2
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color

# Function to generate a mask with a small square in the middle
def generate_square_mask(image, square_size=50):
    mask = np.zeros(image.shape, dtype=bool)
    center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
    half_size = square_size // 2
    mask[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = True
    return mask


# Parameters
lambda_values = np.linspace(1.0, 10, 10)  # 10 values between 0.01 and 0.1
T = 2500
dt = 0.5

best_psnr = 0
best_mse = float('inf')
best_lambda = 0

# Load the images
original = io.imread(r"C:\Users\kizer\Master_code\input_0_zoom.png")
gray_image = color.rgb2gray(original)

# Generate a mask with a small square in the middle
mask = generate_square_mask(gray_image, square_size=20)

# Apply the mask to the grayscale image (set masked areas to 0)
masked_image = np.copy(gray_image)
masked_image[mask] = 0

# Set up progress bars for lambda iterations
for lambda_val in tqdm(lambda_values, desc='Lambda Iterations'):
    # Inpaint the image
    inpainted_image, _ = TV(masked_image, lambda_val, mask, T, dt)
    
    # Calculate PSNR
    psnr_value = cv2.PSNR(gray_image, inpainted_image)
    
    # Calculate MSE
    mse_value = mse(gray_image, inpainted_image)
    
    # Check if this combination is better
    if psnr_value > best_psnr or (psnr_value == best_psnr and mse_value < best_mse):
        best_psnr = psnr_value
        best_mse = mse_value
        best_lambda = lambda_val
        print(f"New Best PSNR: {best_psnr} dB, MSE: {best_mse}, Lambda: {best_lambda}")

# Print the best parameters found
print(f"Best Lambda: {best_lambda}")
print(f"Best PSNR: {best_psnr} dB, Best MSE: {best_mse}")

# Optionally, visualize the best inpainting result
best_inpainted_image, _ = TV(masked_image, best_lambda, mask, T, dt)
best_inpainted_image_normalized = (best_inpainted_image - np.min(best_inpainted_image)) / (np.max(best_inpainted_image) - np.min(best_inpainted_image))
best_inpainted_image_uint8 = img_as_ubyte(best_inpainted_image_normalized)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(masked_image, cmap='gray')
axes[0].set_title('Masked Image')
axes[0].axis('off')

axes[1].imshow(best_inpainted_image_uint8, cmap='gray')
axes[1].set_title('Best Inpainted Image')
axes[1].axis('off')

plt.show()