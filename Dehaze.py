# By 11713016 Zhihua Xiao
import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.lib.stride_tricks import as_strided
import json


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    '''引导滤波，参考matlab'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Parameter_estimate(m, r, eps, w, maxV1):  #输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  #得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 3), r, eps)  #使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  #计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  #范围限制
    # print('V1 shape', V1.shape)
    return V1, A


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def deHaze(m, r=80, eps=10**(-7), w=0.9, maxV1=0.95):
    Y = np.zeros(m.shape)
    V1, A = Parameter_estimate(m, r, eps, w, maxV1)
    cv2.imwrite('{}_Trans_map.jpg'.format(filename), V1 * 255)
    alpha = V1
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - alpha) / (1 - alpha / A)  #大气退化模型除雾
        # Y[:, :, k] = (m[:, :, k] - A) / alpha + A
    Y = np.clip(Y, 0, 1)
    return Y


def singleScaleRetinex(img, sigma):

    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))

    return retinex


def multiScaleRetinex(img, sigma_list):

    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def simplestColorBalance(img, low_clip, high_clip):

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCP(img, sigma_list, low_clip, high_clip):

    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
        255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


if __name__ == '__main__':
    filename = 'test'
    pictype = '.jpg'
    m = deHaze(cv2.imread('{}{}'.format(filename, pictype)) / 255.0) * 255
    cv2.imwrite('{}_de.jpg'.format(filename), m)
    with open('config.json', 'r') as f:
        config = json.load(f)
        img = m
        print('msrcp processing......')
        img_msrcp = MSRCP(img, config['sigma_list'], config['low_clip'],
                          config['high_clip'])
        cv2.imwrite('{}_MSRCP.jpg'.format(filename), img_msrcp)
