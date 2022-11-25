import cv2 as cv
import numpy as np
import scipy.ndimage
import scipy.io as sio
import matlab
import matlab.engine
from PIL import Image
import numpy as np

eng = matlab.engine.start_matlab() # 启动 MATLAB 引擎

chip_input = cv.imread("chip_input.png")
path = "chip_input.png"
chip = Image.open(path)
rgb = chip.convert("RGB")
img = np.array(rgb)
# RGB original image
chip_4 = cv.resize(chip_input,(0,0), fx=4, fy=4)
cv.imshow("Original image",chip_4)

# RGB to YCR_CB
mat = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])

def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape)
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img

img = rgb2ycbcr(img)
y_channel = img[:,:,0]
u_channel = img[:,:,1]
v_channel = img[:,:,2]

# Bicubic Upsample
[H_y,H_u,H_v,H_image] = eng.getbicubic(nargout=4)

load_H_y = 'H_tilde_Y.mat'
load_data = sio.loadmat(load_H_y)
H_y = load_data['H_tilde_Y']

load_H_Cb = 'H_Cb.mat'
load_data = sio.loadmat(load_H_Cb)
H_u = load_data['H_Cb']

load_H_Cr = 'H_Cr.mat'
load_data = sio.loadmat(load_H_Cr)
H_v = load_data['H_Cr']

load_H_bicubic = 'H_bicubic.mat'
load_data = sio.loadmat(load_H_bicubic)
H_image = load_data['H_bicubic']
cv.imshow("H_image", H_image)

# Feed-back control loop
DECONV = {'lambda_1': 0.01, 'lambda_2': 5.0, 'iter_max': 10}
GAU = {'size': 11.0, 'var': 1.5}

for i in range(1,5):
    # Normalize
    [H_y_normal, low, gap] = eng.simpnormimg(matlab.double(H_y.tolist()), nargout=3);
    # Non-blind deconvolution
    H_y_star = eng.nbDeconv(matlab.double(H_y_normal), DECONV, GAU);
    H_y_star = np.array(H_y_star)
    H_y_star = H_y_star * gap + low;
    if (i == 4):
        break
    # Reconvolution
    c = 3
    r = 3
    sigma = 1
    PSF_kernel = np.multiply(cv.getGaussianKernel(r, sigma), (cv.getGaussianKernel(c, sigma)).T)
    H_y_s = cv.filter2D(H_y_star.astype('float32'), -1, PSF_kernel, borderType=cv.BORDER_CONSTANT)
    # Pixel substitution
    H_y = H_y_s
    for i in range(0,y_channel.shape[0]):
        for j in range(0,y_channel.shape[1]):
            H_y[i*4][j*4] = y_channel[i][j] / 255

H_star = cv.merge([H_y_star,H_u,H_v])
print(H_star[:,:,0])
print(H_star[:,:,0].shape)
input_out = H_star * 255
output1 = input_out.astype(np.uint8)  # python类型转换
output2 = cv.cvtColor(output1, cv.COLOR_YCR_CB2RGB)
print(output2)
cv.imshow("High resolution",output2)
cv.imwrite("High resolution1.jpg",output2)
cv.waitKey(0)
cv.destroyAllWindows()