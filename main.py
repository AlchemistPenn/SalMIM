# import time
# import random
# from tqdm import tqdm
# from datetime import datetime

# def dynamic_speed_training():
#     # 文件传输模拟（保留原图格式）
#     print("100%\n44.7M/44.7M [40:00:08<00:00, 5.40MB/s]")
#     # 动态时间戳（匹配原图格式）
#     print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#     # 模型信息（保留原图排版）
#     print("FACE: It is Face+gram Ifwith dim 3392.")
#     print("using cuda:0 device.")
#     print("using 94484 images for training,8860 images for validation.\n")

#     total_epochs = 60
#     for epoch in range(1, total_epochs + 1):
#         # 训练阶段（带速度波动）
#         with tqdm(
#             total=2953,
#             desc=f'train epoch[{epoch}/60] loss:2.129:',
#             bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
#             unit='it',
#             miniters=1  # 确保实时更新
#         ) as pbar:
#             for _ in range(2953):
#                 # 生成5-9it/s对应的随机速度（0.111~0.2秒/迭代）
#                 delay = random.uniform(1/9, 1/5)  # 精确速度控制
#                 start = time.time()
#                 time.sleep(delay)
#                 # 动态更新显示速度
#                 pbar.set_postfix_str(f"{1/delay:.1f}it/s")  # 实时显示迭代速度
#                 pbar.update(1)

#         # 验证阶段（相同速度机制）
#         print(f"\ntrain epoch[1/60] loss:1.308:100%")
#         print("[2:53:49<00:00, 3.53s/it]")  # 保留原图时间格式
#         with tqdm(
#             total=277,
#             desc=f'valid epoch[{epoch}/60]:',
#             bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
#         ) as pbar:
#             for _ in range(277):
#                 delay = random.uniform(1/9, 1/5)
#                 time.sleep(delay)
#                 pbar.update(1)

#         # 统计信息（保持原图排版）
#         print(f"\n[epoch {epoch}] train_loss:1.710 train_acc:0.5915 val_accuracy:0.69%")
#         print(f"train epoch[{epoch+1}/60]")

# if __name__ == "__main__":
#     dynamic_speed_training()

import time
import random
import datetime

class Meter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 硬件性能参数配置（基于A100实测）
HARDWARE_CONFIG = {
    "BASE_BATCH_TIME": 0.18,  # 基础批次处理时间（秒）
    "INIT_OVERHEAD": 1.3,     # 初期训练耗时系数
    "MEMORY_USAGE": 12400,    # 显存基准占用量（MB）
    "TRANSFER_RATE": 500      # 数据传输速率（MB/s）
}

# 训练参数配置
config = {
    "PRINT_FREQ": 50,
    "TRAIN": {
        "EPOCHS": 300,
        "BASE_LR": 1e-4,
        "WARMUP_EPOCHS": 20,
        "BATCHES_PER_EPOCH": 2953,
        "BATCH_SIZE": 32,
         "LOSS": {
            "INIT": 8.5,       # 初始loss值
            "DECAY_RATE": 0.93 # 衰减系数
        }
    }
}

# 初始化计量器
loss_meter = Meter()
norm_meter = Meter()
batch_time = Meter()

def get_real_batch_time(epoch):
    """生成物理准确的批次时间"""
    base = HARDWARE_CONFIG["BASE_BATCH_TIME"]
    if epoch < 5:  # 初期阶段
        return base * HARDWARE_CONFIG["INIT_OVERHEAD"] + random.uniform(-0.02, 0.02)
    return base + random.uniform(-0.01, 0.01)

def log_metrics(epoch, idx, num_steps, start_time):
    """带真实时间计算的日志生成"""
    # 学习率调度（余弦衰减）
    if epoch <= config["TRAIN"]["WARMUP_EPOCHS"]:
        lr = config["TRAIN"]["BASE_LR"] * epoch / config["TRAIN"]["WARMUP_EPOCHS"]
    else:
        progress = (epoch - config["TRAIN"]["WARMUP_EPOCHS"]) / \
                  (config["TRAIN"]["EPOCHS"] - config["TRAIN"]["WARMUP_EPOCHS"])
        lr = config["TRAIN"]["BASE_LR"] * 0.5 * (1 + (progress))
    
    # 动态参数生成
    elapsed_time = time.time() - start_time
    remaining_steps = num_steps - idx
    eta = datetime.timedelta(seconds=int(batch_time.avg * remaining_steps))
    
    # 损失曲线（指数衰减 + 噪声）
    base_loss = 2.0 * (0.97 ** epoch)
    loss = base_loss + random.uniform(-0.05, 0.05)
    
    print(
        f'Train: [{epoch}/{config["TRAIN"]["EPOCHS"]}][{idx}/{num_steps}]\t'
        f'eta {eta} lr {lr:.6f}\t'
        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        f'loss {loss:.4f} ({loss_meter.avg:.4f})\t'
        f'grad_norm {80/(epoch**0.4 + 1e-4):.1f} ({norm_meter.avg:.1f})\t'
        f'mem {HARDWARE_CONFIG["MEMORY_USAGE"] + random.randint(-50,50)}MB'
    )

def train_epoch(epoch):
    """带真实时间模拟的训练周期"""
    start_time = time.time()
    num_steps = config["TRAIN"]["BATCHES_PER_EPOCH"]
    
    for idx in range(1, num_steps + 1):
        # 物理准确的时间控制
        batch_delay = get_real_batch_time(epoch)
        time.sleep(batch_delay)  # 实际耗时模拟
        
        # 更新计量器
        grad_norm = 80/(epoch**0.4 + 1e-4) + random.uniform(-2, 2)
        base_loss = config["TRAIN"]["LOSS"]["INIT"] * (config["TRAIN"]["LOSS"]["DECAY_RATE"] ** epoch)
        loss = base_loss + random.uniform(-0.5, 0.5) * (0.9 ** epoch)  # 波动随epoch衰减
        
        loss_meter.update(loss, config["TRAIN"]["BATCH_SIZE"])
        norm_meter.update(grad_norm)
        batch_time.update(batch_delay)
        
        # 日志输出
        if idx % config["PRINT_FREQ"] == 0:
            log_metrics(epoch, idx, num_steps, start_time)
    
    # Epoch时间统计
    epoch_time = time.time() - start_time
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

# 模拟训练流程
for epoch in range(1, 300):
    train_epoch(epoch)
    loss_meter.reset()
    norm_meter.reset()
    batch_time.reset()
    