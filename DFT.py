# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 加载图像并转换为灰度图像
# img_path = '/home/pengwy/Data/MyWork/contentment_02271.jpg'  # 替换为你图片的路径
# image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# # 获取图像的傅里叶变换
# f = np.fft.fft2(image)
# fshift = np.fft.fftshift(f)

# # 生成低通滤波器
# rows, cols = image.shape
# crow, ccol = rows // 2 , cols // 2

# # 创建掩码，只保留低频成分
# mask = np.zeros((rows, cols), np.uint8)
# radius = 30  # 低频保留的半径，可根据需求调整
# cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

# # 应用掩码
# fshift_filtered = fshift * mask

# # 进行逆傅里叶变换以获得滤波后的图像
# f_ishift = np.fft.ifftshift(fshift_filtered)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)

# # 保存滤波后的图像
# output_path = '/home/pengwy/Data/SimMIM/low_frequency_image.jpg'  # 替换为你想保存的路径
# cv2.imwrite(output_path, img_back)

# print(f"Filtered image saved to: {output_path}") 


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像并转换为彩色图像
img_path = '/home/pengwy/Data/MyWork/U2535P461T5D228548F154DT20080819024018.jpg'  # 替换为你图片的路径
image = cv2.imread(img_path)

# 分离图像的颜色通道
b, g, r = cv2.split(image)

# 定义一个函数来处理每个通道
def low_pass_filter(channel, radius):
    # 获取图像的傅里叶变换
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    # 生成低通滤波器
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # 创建掩码，只保留低频成分
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

    # 应用掩码
    fshift_filtered = fshift * mask

    # 进行逆傅里叶变换以获得滤波后的图像
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

# 对每个颜色通道应用低通滤波器
radius = 30  # 低频保留的半径，可根据需求调整
b_filtered = low_pass_filter(b, radius)
g_filtered = low_pass_filter(g, radius)
r_filtered = low_pass_filter(r, radius)

# 合并处理后的颜色通道
filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))

# 将结果转换为uint8类型
filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

# 保存滤波后的图像
output_path = '/home/pengwy/Data/SimMIM/low_frequency_image1.jpg' # 替换为你想保存的路径
cv2.imwrite(output_path, filtered_image)

print(f"Filtered image saved to: {output_path}")