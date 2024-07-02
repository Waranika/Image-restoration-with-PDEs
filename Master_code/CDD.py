import numpy as np
import scipy.signal
import math

def compute_curvature(u):
    """
    Compute the curvature of the image u.
    """
    # Compute gradients (creating arrays filled with gradients of the image values in each direction)
    u_x = np.gradient(u, axis=1)
    u_y = np.gradient(u, axis=0)

    # Compute gradient magnitude 
    magnitude = np.sqrt(u_x**2 + u_y**2) + 1e-8  # Added small constant to avoid division by zero

    # Compute curvature
    curvature = np.gradient(u_x/magnitude, axis=1) + np.gradient(u_y/magnitude, axis=0) 
    return curvature

def cdd_inpainting(image, mask, g, iterations=100, tau=0.1):
    """
    Performs Curvature-Driven Diffusions (CDD) inpainting on a given image.
    
    Parameters:
        image: 2D numpy array representing the grayscale image.
        mask: 2D boolean numpy array where True indicates missing pixels to inpaint.
        g: Function that modifies diffusion based on curvature.
        iterations: Number of iterations to run the inpainting process.
        tau: Time step size.
        
    Returns:
        Inpainted image as a 2D numpy array.
    """
    u = image.astype(np.float64)  # Convert image to float for processing
    
    for iter_num in range(iterations):
        curvature = compute_curvature(u)
        # Compute gradient magnitude 
        magnitude = np.sqrt(np.gradient(u, axis=1)**2 + np.gradient(u, axis=0)**2) +1e-8
        D = g(np.abs(curvature))/magnitude

        # Compute j , the flux field
        j_x = -D*np.gradient(u, axis=1)
        j_y = -D*np.gradient(u, axis=0)
        
        # Compute the divergence
        divergence = np.gradient(j_x, axis=1) + np.gradient(j_y, axis=0)
        
        # Update only the inpainting domain
        u[mask] -= tau * divergence[mask]
        
        # Debugging: print the iteration number and max divergence
        if iter_num % 100 == 0 or iter_num == iterations - 1:
            print(f"Iteration {iter_num + 1}/{iterations}, max divergence: {np.max(divergence)}")
    
    # Convert the processed image back to the original type
    u = np.clip(u, 0, 255)  # Clip values to be in the valid range
    u = u.astype(image.dtype)
    return u
