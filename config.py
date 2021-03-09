import torch

batch_size = 512  # 训练的批次
train_device = 'cuda:0'  # 选择训练设备,可选项'cuda device, i.e. 0 or 0,1,2,3 or cpu'
test_device = 'cuda:0'  # 选择使用设备,可选项'cuda device, i.e. 0 or 0,1,2,3 or cpu'
image_size = (100, 32)  # 图片尺寸(w, h)
max_seq_len = 15  # 最大序列长度
epoch = 1000  # 总计训练轮次
num_workers = 4  # 训练中加载数据的核心数
net_weights = 'weights'  # 网络权重保存地址
data = '/media/cq/data/public/hibiki/CCPD2019/data/chepai'  # 训练数据地址
is_continue_training = False  # 是否继续上一次的结果训练
accumulation_steps = 1  # 是否使用梯度累计,
load_weight = 'best'  # 加载权重的模式，best or last
logs = 'weights'  # 日志文件保存地址
learning_rate = 1e-3  # 学习率


"""类别名称"""
class_names = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '*']
blank_char = '*'

"""数据增强选项"""
salt_noise = True  # 是否添加椒盐噪声
salt_noise_probs = 0.02  # 椒盐噪声的比例
salt_noise_probability = 0.5  # 添加椒盐噪声的概率

randomly_adjust_brightness = True  # 是否开启随机明亮度变化
randomly_adjust_brightness_probability = 0.5  # 随机明亮度变化的概率

random_cutting = True  # 是否随机裁切
random_cutting_probability = 0.5  #随机裁切的概率
random_cutting_size = 3

# 随机色彩变化配置
to_hsv = True  # 是否开始色彩空间转换增强
to_hsv_probs = 0.5  # 色彩空间增强的概率

# 高斯模糊配置
gauss_blur = True  # 是否开启随机高斯模糊
gauss_blur_probs = 0.5  # 高斯模糊的概率
gauss_blur_max_level = 6  # 高斯的最高模糊等级
# 随机空间扭曲配置
random_space_distortion = True  # 是否开启随机空间扭曲
distortion_probs = 0.5  # 空间扭曲的概率


"""根据配置计算的参数"""
num_class = len(class_names)
train_device = train_device if torch.cuda.is_available() else "cpu"
test_device = test_device if torch.cuda.is_available() else "cpu"

