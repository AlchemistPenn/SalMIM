import cv2
import numpy as np

# 加载图像
# image_path = '/home/pengwy/Data/MyWork/U2535P461T5D228548F154DT20080819024018.jpg'  # 替换为你的图像路径
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # 图像大小
# rows, cols = image.shape
# crow, ccol = rows // 2, cols // 2  # 中心位置

# # 傅里叶变换
# dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

# # 创建不同带通半径的滤波器
# radius_list = [30, 60, 100]  # 中心半径
# for i, radius in enumerate(radius_list):
#     mask = np.ones((rows, cols, 2), np.float32)

#     # 中心区域更强的衰减
#     cv2.circle(mask, (ccol, crow), radius, (0.5, 0.5), -1)  # 中心区域
#     cv2.rectangle(mask, (ccol - 50, crow - 50), (ccol + 50, crow + 50), (0.2, 0.2), -1)  # 小区域衰减更强

#     # 频谱过滤
#     filtered_dft = dft_shift * mask

#     # 逆变换到空间域
#     filtered_idft_shift = np.fft.ifftshift(filtered_dft)
#     filtered_image = cv2.idft(filtered_idft_shift)
#     filtered_image_magnitude = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])

#     # 计算频谱图
#     magnitude_spectrum = 20 * np.log(cv2.magnitude(filtered_dft[:, :, 0], filtered_dft[:, :, 1]) + 1)

#     # 保存频谱图
#     spectrum_path = f'/home/pengwy/Data/SimMIM/filtered_spectrum_R{radius}.png'
#     cv2.imwrite(spectrum_path, magnitude_spectrum.astype(np.uint8))
#     print(f"频谱图已保存: {spectrum_path}")

#     # # 保存滤波后的图像
#     # filtered_image_path = f'filtered_image_R{radius}.png'
#     # cv2.imwrite(filtered_image_path, filtered_image_magnitude.astype(np.uint8))
#     # print(f"滤波后的图像已保存: {filtered_image_path}")


import cv2
import numpy as np

# 加载原图和 saliency map
original_img = cv2.imread('/home/pengwy/Data/MyWork/U2535P461T5D228548F154DT20080819024018.jpg')  # 替换为原图路径
saliency_map = cv2.imread('/home/pengwy/Data/TranSalNet/example/result1.png', cv2.IMREAD_GRAYSCALE)  # 替换为 Saliency Map 路径，读取为单通道

# 二值化处理
_, binary_map = cv2.threshold(saliency_map, 50, 255, cv2.THRESH_BINARY)  # 阈值 50，可根据需要调整

# 查找轮廓
contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 合并所有轮廓点
all_points = np.concatenate(contours)  # 将所有轮廓点整合为一个数组

# 获取包围所有高亮区域的最小矩形框
x, y, w, h = cv2.boundingRect(all_points)

# 创建掩码后的图像
masked_inside = original_img.copy()  # 框内掩码
masked_outside = original_img.copy()  # 框外掩码

# 框内掩码：把框内区域用黑色覆盖，其余保持不变
masked_inside[y:y+h, x:x+w] = 0

# 框外掩码：把框外区域用黑色覆盖，其余保持不变
outside_mask = np.zeros_like(original_img, dtype=np.uint8)
outside_mask[y:y+h, x:x+w] = original_img[y:y+h, x:x+w]
masked_outside = outside_mask

# 保存结果
inside_mask_path = '/home/pengwy/Data/MyWork/masked_inside.jpg'
outside_mask_path = '/home/pengwy/Data/MyWork/masked_outside.jpg'

cv2.imwrite(inside_mask_path, masked_inside)
cv2.imwrite(outside_mask_path, masked_outside)

print(f"框内掩码已保存到 {inside_mask_path}")
print(f"框外掩码已保存到 {outside_mask_path}")

