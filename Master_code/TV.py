import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

def TV(input_img, lambda_val, mask, T, dt):
    i, j = input_img.shape
    mask = (mask == 0).astype(float)
    input_img = input_img.astype(float)
    input_img = mask * input_img 
    u = input_img.copy()
    iterations = 0

    plt.figure()
    plt.title('Inpainting...')
    plt.imshow(u, cmap='gray')
    plt.show(block=False)

    
    for t in np.arange(0, T, dt):
        u_x = np.gradient(u, axis=1)
        u_y = np.gradient(u, axis=0)
        N = np.sum(np.sqrt(u_x**2 + u_y**2))

        u_xx = np.gradient(u_x, axis=1)
        u_yy = np.gradient(u_y, axis=0)
        u_xy = np.gradient(u_x, axis=0)

        deltaE = -(u_xx * u_y**2 - 2*u_x * u_y * u_xy + u_yy * u_x**2) / (1e-5  + 0.1 + (u_x**2 + u_y**2)**(3/2)) + 2 * mask * (lambda_val * (u - input_img))

        u = dt * (-deltaE ) + u 
        #print(N)
        iterations += 1

        if iterations % 100 == 0:
            plt.clf()
            plt.imshow(u / 255, cmap='gray')
            plt.title(f'Inpainting: {iterations} iterations...')
            plt.draw()
            plt.pause(0.001)

    

    return u, N