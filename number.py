import numpy as np

# 设置均值和标准偏差
mean = 81.04
std_dev = 0.17

num_samples = 10

# 生成符合正态分布的样本
samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
rounded_samples = [round(sample, 2) for sample in samples]

# 输出生成的数字
print(rounded_samples)

# 输出生成的数字
# print(samples)
 