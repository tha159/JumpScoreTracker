import cv2
import numpy as np

# 读取图像
image = cv2.imread('../img/ruler.jpg')

# 物体的实际尺寸和图像中的像素尺寸（这里假设宽度为100个像素）
w_real = 600  # 假设物体的实际宽度为10厘米
w_pixel = 1100  # 假设图像中物体的宽度为100像素
pixel_width = image.shape[1]  # 图像的宽度（像素单位）

# 估计相机焦距
f_estimated = (w_real / w_pixel) * pixel_width

# 手动设置相机参数
camera_matrix = np.array([[f_estimated, 0, image.shape[1] / 2],
                          [0, f_estimated, image.shape[0] / 2],
                          [0, 0, 1]], dtype=np.float32)

# 畸变校正
undistorted_image = cv2.undistort(image, camera_matrix, None)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
