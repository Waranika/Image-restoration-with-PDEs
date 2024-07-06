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

    for t in tqdm(np.arange(0, T + dt, dt)):
        u_x = np.roll(u, -1, axis=1) - u
        u_y = np.roll(u, -1, axis=0) - u
        N = np.sum(np.sqrt(u_x**2 + u_y**2))

        u_xx = np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)
        u_yy = np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)
        u_xy = (np.roll(np.roll(u, -1, axis=0), -1, axis=1) + 
                np.roll(np.roll(u, 1, axis=0), 1, axis=1) - 
                np.roll(np.roll(u, 1, axis=0), -1, axis=1) - 
                np.roll(np.roll(u, -1, axis=0), 1, axis=1)) / 4

        deltaE = -(u_xx * u_y**2 - 2*u_x * u_y * u_xy + u_yy * u_x**2) / (0.1 + (u_x**2 + u_y**2)**(3/2)) + \
                 2 * mask * (lambda_val * (u - input_img))

        u = dt * (-deltaE * N / (np.sqrt(np.sum(N**2)))) + u
        iterations += 1

        if iterations % 100 == 0:
            plt.clf()
            plt.imshow(u / 255, cmap='gray')
            plt.title(f'Inpainting: {iterations} iterations...')
            plt.draw()
            plt.pause(0.001)

    plt.clf()
    plt.imshow(u.astype(np.uint8), cmap='gray')
    plt.title(f'Done after {iterations} iterations!')
    plt.show()

    return u.astype(np.uint8), N