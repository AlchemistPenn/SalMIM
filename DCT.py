import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# 加载图像并转换为彩色图像
img_path = '/home/pengwy/Data/MyWork/contentment_02271.jpg'  # 替换为你图片的路径
image = cv2.imread(img_path)

# 分离图像的颜色通道
b, g, r = cv2.split(image)

# 定义一个函数来处理每个通道
def low_pass_filter_dct(channel, keep_fraction):
    # 获取图像的离散余弦变换
    dct_transformed = dct(dct(channel.T, norm='ortho').T, norm='ortho')

    # 创建掩码，只保留低频成分
    rows, cols = channel.shape
    mask = np.zeros((rows, cols))
    mask[:int(rows * keep_fraction), :int(cols * keep_fraction)] = 1

    # 应用掩码
    dct_filtered = dct_transformed * mask

    # 进行逆离散余弦变换以获得滤波后的图像
    img_back = idct(idct(dct_filtered.T, norm='ortho').T, norm='ortho')
    img_back = np.abs(img_back)

    return img_back

# 对每个颜色通道应用低通滤波器
keep_fraction = 0.1  # 保留低频的比例，可根据需求调整
b_filtered = low_pass_filter_dct(b, keep_fraction)
g_filtered = low_pass_filter_dct(g, keep_fraction)
r_filtered = low_pass_filter_dct(r, keep_fraction)

# 合并处理后的颜色通道
filtered_image = cv2.merge((b_filtered, g_filtered, r_filtered))

# 将结果转换为uint8类型
filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

# 保存滤波后的图像
output_path = '/home/pengwy/Data/SimMIM/DCT_image.jpg'  # 替换为你想保存的路径
cv2.imwrite(output_path, filtered_image)

print(f"Filtered image saved to: {output_path}")