from torch.utils.data import DataLoader
from torch import nn
from utils.ocrdataset import OcrDataset
from tqdm import tqdm
from ocr_models.net import OcrNet
import torch
import os
import config


class Trainer:

    def __init__(self):
        self.device = torch.device(config.train_device)
        print('准备使用设备%s训练网络' % self.device)
        self.net = OcrNet(config.num_class)
        if os.path.exists(f'{config.net_weights}/net.pt') and config.is_continue_training:
            self.net.load_state_dict(torch.load(f'{config.net_weights}/net.pt', map_location='cpu'))
            print('成功加载模型参数')
        elif not config.is_continue_training:
            print('未能加载模型参数')
        else:
            raise RuntimeError('Model parameters are not loaded')
        '''训练集加载器'''
        self.dataset = OcrDataset()
        self.dataloader = DataLoader(self.dataset, config.batch_size, True)

        self.net = self.net.to(self.device)
        self.loss_func = nn.CTCLoss(blank=config.num_class - 1, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)

    def __call__(self):
        epoch = 1
        count = 0
        while True:

            '''训练过程'''
            loss_sum = 0
            for images, targets, target_lengths in tqdm(self.dataloader):
                images = images.to(self.device)

                '''生成标签'''
                e = torch.tensor([])
                for i, j in enumerate(target_lengths):
                    e = torch.cat((e, targets[i][:j]), dim=0)
                targets = e.long()
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                '''预测'''
                predict = self.net(images)
                s, n, v = predict.shape
                input_lengths = torch.full(size=(n,), fill_value=s, dtype=torch.long)

                """计算损失，预测值需，log_softmax处理，网络部分不应该softmax"""
                loss = self.loss_func(predict.log_softmax(2), targets, input_lengths, target_lengths)

                '''反向传播，梯度更新'''
                loss.backward()
                if (count + 1) % config.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                '''统计日志信息'''
                loss_sum += loss.item()
                count += 1
            logs = f'''epoch:{epoch}, loss_sum: {loss_sum / len(self.dataloader)}'''
            print(logs)
            torch.save(self.net.state_dict(), f'{config.net_weights}/ocr_net.pt')
            epoch += 1


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
