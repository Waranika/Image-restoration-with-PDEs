import math
import skimage
import numpy as np
import scipy.signal
from skimage.util import random_noise

# Function to generate a mask with random noise
def generate_mask(image, noise_amount=0.2):
    noisy_image = random_noise(image, mode='s&p', amount=noise_amount)
    mask = noisy_image != image
    return mask


def nonlinearDiffusionFilter(image: np.ndarray, iterations=5, lamb=1.0, tau=0.125, image_seq=None):
    """
    Execute nonlinear isotropic smoothing filter on an image.
    The method is based on the 1990 paper by Perona and Malik.
    This smoothing method uses diffusion that preserves edges.
    """
    def computeUpdate(u: np.ndarray, g: np.ndarray):
        """
        Compute the update for the next iteration using spatial derivatives.
        """
        update = np.zeros(u.shape, dtype=float)
        u_padded = np.pad(u, pad_width=1, mode='constant')
        g_padded = np.pad(g, pad_width=1, mode='constant')

        for i in range(1, u_padded.shape[1]-1):
            for j in range(1, u_padded.shape[0]-1):
                g_pj = math.sqrt(g_padded[j, i+1] * g_padded[j, i])
                g_nj = math.sqrt(g_padded[j, i-1] * g_padded[j, i])
                g_ip = math.sqrt(g_padded[j+1, i] * g_padded[j, i])
                g_in = math.sqrt(g_padded[j-1, i] * g_padded[j, i])

                if i==u.shape[1]-2:
                    g_pj = 0
                if i==1:
                    g_nj = 0
                if j==u.shape[0]-2:
                    g_ip = 0
                if j==1:
                    g_in = 0

                ux0 =  g_pj * (u_padded[j, i+1] - u_padded[j, i])
                ux1 = -g_nj * (u_padded[j, i] - u_padded[j, i-1])
                uy0 =  g_ip * (u_padded[j+1, i] - u_padded[j, i])
                uy1 = -g_in * (u_padded[j, i] - u_padded[j-1, i])

                # Update is not padded, so subtract 1 from indices
                update[j-1, i-1] = ux0 + ux1 + uy0 + uy1

        return update

    def computeDiffusivity(u: np.ndarray, lamb: float):
        """
        Compute the nonlinear gradient-derived diffusivity.
        """
        shape = u.shape
        if len(shape) > 2 and shape[2] > 1:
            print("RGB to gray")
            u = skimage.color.rgb2gray(u)
    
        gradkernelx = 0.5 * np.array([[0.0, 0.0, 0.0], 
                                      [-1.0, 0.0, 1.0], 
                                      [0.0, 0.0, 0.0]])
        gradkernely = 0.5 * np.array([[0.0, -1.0, 0.0], 
                                      [0.0, 0.0, 0.0], 
                                      [0.0, 1.0, 0.0]])
        gradx = scipy.signal.convolve2d(u, gradkernelx, boundary='symm')
        grady = scipy.signal.convolve2d(u, gradkernely, boundary='symm')
        gradm2 = np.square(gradx) + np.square(grady)
        g = 1.0 / (np.sqrt((1.0 + gradm2) / (lamb*lamb)))
        return g

    u = np.copy(image)
    if len(u.shape) > 2 and u.shape[2] == 1:
        u = np.reshape (u, u.shape[0], u.shape[1])
    if image_seq != None:
        image_seq.append(np.copy(u))

    for i in range(iterations):
        print(f"Iterations: {i+1}/{iterations}")
        g = computeDiffusivity(u, lamb)
        update = computeUpdate(u, g)
        u += tau * update
        if image_seq != None:
            image_seq.append(np.copy(u))
    return u
