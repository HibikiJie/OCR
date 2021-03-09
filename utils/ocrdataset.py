from torch.utils.data import Dataset
from utils import enhance
import torch
import numpy
import cv2
import os
import random
import config


class OcrDataset(Dataset):

    def __init__(self, *, is_train=True):
        super(OcrDataset, self).__init__()
        self.is_train = is_train
        self.dataset = []
        root = config.data
        for image_name in os.listdir(root):
            targets = image_name.split('_')[-1][:-4]
            image_path = f'{root}/{image_name}'
            _targets = []
            for target in targets:
                _targets.append(config.class_names.index(target))
            self.dataset.append([image_path, _targets])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, target = self.dataset[item]
        image = cv2.imread(image_path)

        """应用数据增强"""
        image = self.data_to_enhance(image)

        """转换数据格式，为tensor"""
        # image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image).permute(2, 0, 1) / 255
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        _target = torch.full(size=(config.max_seq_len,), fill_value=0, dtype=torch.long)
        _target[:len(target)] = target
        # print(image.shape)
        return image, _target, target_length

    def data_to_enhance(self, image):
        """
        数据增强
            包含图片随机裁剪，随机明亮度变化，随机饱和度变化，随机椒盐噪声，随机色彩变化，高斯模糊，空间扭曲
        :param image: 输入图片
        :return: image: 输出图片
        """
        if self.is_train:
            """图片随机裁剪"""
            if random.random() > config.random_cutting_probability and config.random_cutting:
                image = enhance.random_cutting(image, config.random_cutting_size)

            """随机明亮饱和度变化"""
            if random.random() < config.randomly_adjust_brightness_probability and config.randomly_adjust_brightness:
                image = enhance.randomly_adjust_brightness(image, random.randint(-50, 50), random.randint(-50, 50))

            '''随机椒盐噪声'''
            if random.random() < config.salt_noise_probability and config.randomly_adjust_brightness:
                image = enhance.sp_noise(image, config.salt_noise_probs)

            '''随机hsv色彩变化'''
            if random.random() < config.to_hsv_probs and config.to_hsv:
                if random.random() > 0.5:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            """随机高斯模糊"""
            if random.random() < config.gauss_blur_probs and config.gauss_blur_probs:
                image = enhance.gauss_blur(image, config.gauss_blur_max_level)

            """随机空间扭曲"""
            image, points = enhance.augment_sample(image, max(config.image_size) * 5)
            image = enhance.reconstruct_image(image, [numpy.array(points).reshape((2, 4))],
                                                  config.image_size)[0]
        else:
            image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_AREA)

        # cv2.imshow('a', image)
        # cv2.waitKey(0)
        return image


if __name__ == '__main__':
    d = OcrDataset()
    for i in range(1000):
        d[i]
