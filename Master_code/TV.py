import numpy as np
from scipy.sparse import spdiags
from scipy.ndimage import convolve

def u_grad(u, D1, D2):
    x_grad = np.dot(u, D1)
    y_grad = np.dot(D2, u)
    return x_grad, y_grad

def shrink(x_grad, y_grad, b_x, b_y, gama):
    x1 = (x_grad + b_x) / (np.abs(x_grad + b_x) +1e-8)
    m1 = np.abs(x_grad + b_x) - 1 / (gama + 1e-8) 
    m1 = np.maximum(m1, 0)
    d_x = x1 * m1
    d_x[np.isnan(d_x)] = 0

    x2 = (y_grad + b_y) / (np.abs(y_grad + b_y) +1e-8)
    m2 = np.abs(y_grad + b_y) - 1 / (gama + 1e-8) 
    m2 = np.maximum(m2, 0)
    d_y = x2 * m2
    d_y[np.isnan(d_y)] = 0

    return d_x, d_y

def diff_col(y_data):
    _, col = y_data.shape

    y_data_rowless = np.hstack((y_data[:, 0:1], y_data[:, :-1]))
    y_diff = y_data_rowless - y_data

    return y_diff

def diff_row(x_data):
    row, _ = x_data.shape

    x_data_rowless = np.vstack((x_data[0:1, :], x_data[:-1, :]))
    x_diff = x_data_rowless - x_data

    return x_diff

def gau_sei(d_x, d_y, b_x, b_y, lamda, gama, u, uk, height, width):
    D_prod = lamda * u
    lamda_4g = lamda + 4 * gama

    coef1 = D_prod / (lamda_4g + 1e-8)
    coef2 = gama / (lamda_4g + 1e-8)

    lapla = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    uk_sum = convolve(np.pad(uk, ((1, 1), (1, 1)), 'reflect'), lapla, mode='constant')
    uk_sum = uk_sum[1:height + 1, 1:width + 1]

    db_sum = diff_row(d_x) + diff_col(d_y) - diff_row(b_x) - diff_col(b_y)

    Gk = coef1 + coef2 * (uk_sum + db_sum)
    return Gk

def tv1inpaint(u, lamda, gama, iter, height, width):
    global uk
    uk = np.zeros((height, width))
    b_x = np.zeros((height, width))
    b_y = np.zeros((height, width))
    e1 = np.ones(width)
    e2 = np.ones(height)
    D1 = spdiags([-e1, e1], [0, 1], width, width).toarray()
    D1[width - 1, 0] = 1
    D1 = D1.T
    D2 = spdiags([-e2, e2], [0, 1], height, height).toarray()
    D2[height - 1, 0] = 1

    rmse = np.zeros(iter)

    for num in range(iter):
        x_grad, y_grad = u_grad(uk, D1, D2)
        d_x, d_y = shrink(x_grad, y_grad, b_x, b_y, gama)

        Gk = gau_sei(d_x, d_y, b_x, b_y, lamda, gama, u, uk, height, width)
        dif = (uk - Gk) ** 2
        uk = Gk

        b_x = b_x + x_grad - d_x
        b_y = b_y + y_grad - d_y

        rmse[num] = np.sqrt(np.mean(dif))

    return uk,rmse, num
