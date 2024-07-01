import numpy as np

def TV(input_image, lambda_param, mask, T, dt):
    i, j, _ = input_image.shape
    mask = mask.astype(np.float64)
    input_image = input_image.astype(np.float64)
    input_image = mask * input_image + 255 * (1 - mask) * np.random.rand(*input_image.shape)
    u = input_image.copy()
    iterations = 0

    for t in np.arange(0, T, dt):
        u_x, u_y = np.gradient(u)
        N = np.sum(np.sqrt(u_x**2 + u_y**2))
        u_xx, _ = np.gradient(u_x)
        _, u_yy = np.gradient(u_y)
        u_xy = (np.gradient(u_x, axis=0) + np.gradient(u_y, axis=1)) / 2

        deltaE = -(u_xx * u_y**2 - 2 * u_x * u_y * u_xy + u_yy * u_x**2) / (0.1 + (u_x**2 + u_y**2)**(3/2)) + 2 * mask * (lambda_param * (u - input_image))
        u += dt * (-deltaE * N / (np.sqrt(np.sum(N**2))))

        iterations += 1

    u = u.astype(np.uint8)
    return u, N
