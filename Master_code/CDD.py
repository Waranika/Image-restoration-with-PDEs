import numpy as np

def compute_curvature(u):
    # Approximate gradients
    u_x = np.gradient(u, axis=1)
    u_y = np.gradient(u, axis=0)
    
    # Approximate second-order gradients
    u_xx = np.gradient(u_x, axis=1)
    u_yy = np.gradient(u_y, axis=0)
    u_xy = np.gradient(u_x, axis=0)
    
    # Compute curvature using the formula given in the document
    numerator = u_xx * u_y**2 - 2 * u_x * u_y * u_xy + u_yy * u_x**2
    denominator = (u_x**2 + u_y**2) ** (3 / 2)
    curvature = numerator / (denominator + 1e-8)  # Add a small term to prevent division by zero
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
    u = image.copy()  # Copy of the image to inpaint
    
    for _ in range(iterations):
        curvature = compute_curvature(u)
        diff_coeff = g(np.abs(curvature))
        
        # Compute the divergence of the diffusion coefficient times the gradient
        diff_coeff_grad = np.gradient(diff_coeff)
        u_grad = np.array(np.gradient(u))
        divergence = np.sum(diff_coeff_grad * u_grad, axis=0) + diff_coeff * np.sum(np.gradient(u_grad, axis=1), axis=0)
        
        # Update only the inpainting domain
        u[mask] += tau * divergence[mask]
        
    return u


