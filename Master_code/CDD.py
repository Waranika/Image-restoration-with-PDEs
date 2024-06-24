import numpy as np

def compute_curvature(u):
    """
    Compute the curvature of the image u.
    """
    # Compute gradients
    u_x = np.gradient(u, axis=1)
    u_y = np.gradient(u, axis=0)
    u_xx = np.gradient(u_x, axis=1)
    u_yy = np.gradient(u_y, axis=0)
    u_xy = np.gradient(u_x, axis=0)

    # Compute curvature
    curvature = (u_xx * (1 + u_y**2) - 2 * u_x * u_y * u_xy + u_yy * (1 + u_x**2)) / (1 + u_x**2 + u_y**2)**1.5
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
        diff_coeff = g(np.abs(curvature))
        
        # Compute the divergence of the diffusion coefficient times the gradient
        diff_coeff_grad = np.gradient(diff_coeff)
        u_grad = np.array(np.gradient(u))
        divergence = np.sum(diff_coeff_grad * u_grad, axis=0) + diff_coeff * np.sum(np.gradient(u_grad, axis=1), axis=0)
        
        # Update only the inpainting domain
        u[mask] += tau * divergence[mask]
        
        # Debugging: print the iteration number and max divergence
        if iter_num % 100 == 0 or iter_num == iterations - 1:
            print(f"Iteration {iter_num + 1}/{iterations}, max divergence: {np.max(divergence)}")
    
    # Convert the processed image back to the original type
    u = np.clip(u, 0, 255)  # Clip values to be in the valid range
    u = u.astype(image.dtype)
    return u

