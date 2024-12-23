import math
import numpy as np
import random
from scipy.ndimage.interpolation import shift
from scipy.stats import multivariate_normal


def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).
    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
    Returns:
        ndarray: Rotated sigma matrix.
    """
    D = np.array([[sig_x**2, 0], [0, sig_y**2]])
    U = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return np.dot(U, np.dot(D, U.T))


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.
    Args:
        kernel_size (int):
    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)),
                    yy.reshape(kernel_size * kernel_size,
                               1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.
    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def cdf2(D, grid):
    """Calculate the CDF of the standard bivariate Gaussian distribution.
        Used in skewed Gaussian distribution.
    Args:
        D (ndarrasy): skew matrix.
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.
    Returns:
        cdf (ndarray): skewed cdf.
    """
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    grid = np.dot(grid, D)
    cdf = rv.cdf(grid)
    return cdf


def bivariate_skew_Gaussian(kernel_size, sig_x, sig_y, theta, D, grid=None):
    """Generate a bivariate skew Gaussian kernel.
        Described in `A multivariate skew normal distribution`_ by Shi et. al (2004).
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        D (ndarrasy): skew matrix.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    .. _A multivariate skew normal distribution:
        https://www.sciencedirect.com/science/article/pii/S0047259X03001313
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    pdf = pdf2(sigma_matrix, grid)
    cdf = cdf2(D, grid)
    kernel = pdf * cdf
    kernel = kernel / np.sum(kernel)
    return kernel


def mass_center_shift(kernel_size, kernel):
    """Calculate the shift of the mass center of a kenrel.
    Args:
        kernel_size (int):
        kernel (ndarray): normalized kernel.
    Returns:
        delta_h (float):
        delta_w (float):
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    col_sum, row_sum = np.sum(kernel, axis=0), np.sum(kernel, axis=1)
    delta_h = np.dot(row_sum, ax)
    delta_w = np.dot(col_sum, ax)
    return delta_h, delta_w


def bivariate_skew_Gaussian_center(kernel_size,
                                   sig_x,
                                   sig_y,
                                   theta,
                                   D,
                                   grid=None):
    """Generate a bivariate skew Gaussian kernel at center. Shift with nearest padding.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        D (ndarrasy): skew matrix.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): centered and normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    kernel = bivariate_skew_Gaussian(kernel_size, sig_x, sig_y, theta, D, grid)
    delta_h, delta_w = mass_center_shift(kernel_size, kernel)
    kernel = shift(kernel, [-delta_h, -delta_w], mode='nearest')
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_anisotropic_Gaussian(kernel_size,
                                   sig_x,
                                   sig_y,
                                   theta,
                                   grid=None):
    """Generate a bivariate anisotropic Gaussian kernel.
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_isotropic_Gaussian(kernel_size, sig, grid=None):
    """Generate a bivariate isotropic Gaussian kernel.
    Args:
        kernel_size (int):
        sig (float):
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig**2, 0], [0, sig**2]])
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_Gaussian(kernel_size,
                                   sig_x,
                                   sig_y,
                                   theta,
                                   beta,
                                   grid=None):
    """Generate a bivariate generalized Gaussian kernel.
        Described in `Parameter Estimation For Multivariate Generalized Gaussian Distributions`_
        by Pascal et. al (2013).
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    .. _Parameter Estimation For Multivariate Generalized Gaussian Distributions:
        https://arxiv.org/abs/1302.6498
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(
        -0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau_type1(kernel_size, sig_x, sig_y, theta, beta, grid=None):
    """Generate a plateau-like anisotropic kernel.
    1 / (1+x^(beta))
    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(
        np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau_type1_iso(kernel_size, sig, beta, grid=None):
    """Generate a plateau-like isotropic kernel.
    1 / (1+x^(beta))
    Args:
        kernel_size (int):
        sig (float):
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig**2, 0], [0, sig**2]])
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(
        np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_skew_Gaussian_center(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          noise_range=None,
                                          strict=False):
    """Randomly generate bivariate skew Gaussian kernels at center.
    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])

    sigma_max = np.max([sigma_x, sigma_y])
    thres = 3 / sigma_max
    D = [[np.random.uniform(-thres, thres),
          np.random.uniform(-thres, thres)],
         [np.random.uniform(-thres, thres),
          np.random.uniform(-thres, thres)]]

    kernel = bivariate_skew_Gaussian_center(kernel_size, sigma_x, sigma_y,
                                            rotation, D)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation, D
    else:
        return kernel


def random_bivariate_anisotropic_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          noise_range=None,
                                          strict=False):
    """Randomly generate bivariate anisotropic Gaussian kernels.
    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])

    kernel = bivariate_anisotropic_Gaussian(kernel_size, sigma_x, sigma_y,
                                            rotation)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation
    else:
        return kernel


def random_bivariate_isotropic_Gaussian(kernel_size,
                                        sigma_range,
                                        noise_range=None,
                                        strict=False):
    """Randomly generate bivariate isotropic Gaussian kernels.
    Args:
        kernel_size (int):
        sigma_range (tuple): [0.6, 5]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_range[0] < sigma_range[1], 'Wrong sigma_x_range.'
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    kernel = bivariate_isotropic_Gaussian(kernel_size, sigma)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma
    else:
        return kernel


def random_bivariate_generalized_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          beta_range,
                                          noise_range=None,
                                          strict=False):
    """Randomly generate bivariate generalized Gaussian kernels.
    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y,
                                            rotation, beta)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation, beta
    else:
        return kernel


def random_bivariate_plateau_type1(kernel_size,
                                   sigma_x_range,
                                   sigma_y_range,
                                   rotation_range,
                                   beta_range,
                                   noise_range=None,
                                   strict=False):
    """Randomly generate bivariate plateau type1 kernels.
    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
    assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
    if strict:
        sigma_max = np.max([sigma_x, sigma_y])
        sigma_min = np.min([sigma_x, sigma_y])
        sigma_x, sigma_y = sigma_max, sigma_min
    rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau_type1(kernel_size, sigma_x, sigma_y, rotation,
                                     beta)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma_x, sigma_y, rotation, beta
    else:
        return kernel


def random_bivariate_plateau_type1_iso(kernel_size,
                                       sigma_range,
                                       beta_range,
                                       noise_range=None,
                                       strict=False):
    """Randomly generate bivariate plateau type1 kernels (iso).
    Args:
        kernel_size (int):
        sigma_range (tuple): [0.6, 5]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_range[0] < sigma_range[1], 'Wrong sigma_x_range.'
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])

    kernel = bivariate_plateau_type1_iso(kernel_size, sigma, beta)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    if strict:
        return kernel, sigma, beta
    else:
        return kernel


def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=[0.6, 5],
                         sigma_y_range=[0.6, 5],
                         rotation_range=[-math.pi, math.pi],
                         beta_range=[0.5, 8],
                         noise_range=None):
    """Randomly generate mixed kernels.
    Args:
        kernel_list (tuple): a list name of kenrel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso', 'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise, [0.75, 1.25]. Default: None
    Returns:
        kernel (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_isotropic_Gaussian(
            kernel_size, sigma_x_range, noise_range=noise_range)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_anisotropic_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range)
    elif kernel_type == 'skew':
        kernel = random_bivariate_skew_Gaussian_center(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            noise_range=noise_range)
    elif kernel_type == 'generalized':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_range,
            noise_range=noise_range)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau_type1_iso(
            kernel_size, sigma_x_range, beta_range, noise_range=noise_range)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau_type1(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            beta_range,
            noise_range=noise_range)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(
            noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def show_one_kernel():
    import matplotlib.pyplot as plt
    kernel_size = 21

    # bivariate skew Gaussian
    D = [[0, 0], [0, 0]]
    D = [[3 / 4, 0], [0, 0.5]]
    kernel = bivariate_skew_Gaussian_center(kernel_size, 2, 4, -math.pi / 4, D)
    # bivariate anisotropic Gaussian
    kernel = bivariate_anisotropic_Gaussian(kernel_size, 2, 4, -math.pi / 4)
    # bivariate anisotropic Gaussian
    kernel = bivariate_isotropic_Gaussian(kernel_size, 1)
    # bivariate generalized Gaussian
    kernel = bivariate_generalized_Gaussian(
        kernel_size, 2, 4, -math.pi / 4, beta=4)

    delta_h, delta_w = mass_center_shift(kernel_size, kernel)
    print(delta_h, delta_w)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    # axs.set_axis_off()
    ax = axs[0][0]
    im = ax.matshow(kernel, cmap='jet', origin='upper')
    fig.colorbar(im, ax=ax)

    # image
    ax = axs[0][1]
    kernel_vis = kernel - np.min(kernel)
    kernel_vis = kernel_vis / np.max(kernel_vis) * 255.
    ax.imshow(kernel_vis, interpolation='nearest')

    _, xx, yy = mesh_grid(kernel_size)
    # contour
    ax = axs[1][0]
    CS = ax.contour(xx, yy, kernel, origin='upper')
    ax.clabel(CS, inline=1, fontsize=3)

    # contourf
    ax = axs[1][1]
    kernel = kernel / np.max(kernel)
    p = ax.contourf(
        xx, yy, kernel, origin='upper', levels=np.linspace(-0.05, 1.05, 10))
    fig.colorbar(p)

    plt.show()


# def show_plateau_kernel():
#     import matplotlib.pyplot as plt
#     kernel_size = 21

#     kernel = plateau_type1(kernel_size, 2, 4, -math.pi / 8, 2, grid=None)
#     kernel_norm = bivariate_isotropic_Gaussian(kernel_size, 5)
#     kernel_gau = bivariate_generalized_Gaussian(
#         kernel_size, 2, 4, -math.pi / 8, 2, grid=None)
#     delta_h, delta_w = mass_center_shift(kernel_size, kernel)
#     print(delta_h, delta_w)

    # kernel_slice = kernel[10, :]
    # kernel_gau_slice = kernel_gau[10, :]
    # kernel_norm_slice = kernel_norm[10, :]
    # fig, ax = plt.subplots()
    # t = list(range(1, 22))

    # ax.plot(t, kernel_gau_slice)
    # ax.plot(t, kernel_slice)
    # ax.plot(t, kernel_norm_slice)

    # t = np.arange(0, 10, 0.1)
    # y = np.exp(-0.5 * t)
    # y2 = np.reciprocal(1 + t)
    # print(t.shape)
    # print(y.shape)
    # ax.plot(t, y)
    # ax.plot(t, y2)
    # plt.show()

    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # # axs.set_axis_off()
    # ax = axs[0][0]
    # im = ax.matshow(kernel, cmap='jet', origin='upper')
    # fig.colorbar(im, ax=ax)

    # # image
    # ax = axs[0][1]
    # kernel_vis = kernel - np.min(kernel)
    # kernel_vis = kernel_vis / np.max(kernel_vis) * 255.
    # ax.imshow(kernel_vis, interpolation='nearest')

    # _, xx, yy = mesh_grid(kernel_size)
    # # contour
    # ax = axs[1][0]
    # CS = ax.contour(xx, yy, kernel, origin='upper')
    # ax.clabel(CS, inline=1, fontsize=3)

    # # contourf
    # ax = axs[1][1]
    # kernel = kernel / np.max(kernel)
    # p = ax.contourf(
    #     xx, yy, kernel, origin='upper', levels=np.linspace(-0.05, 1.05, 10))
    # fig.colorbar(p)

    # plt.show()
