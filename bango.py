from ocr_models.net import OcrNet
import torch
import cv2
import config
import os


class Bango:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and config.test_device else 'cpu')
        self.net = OcrNet(config.num_class)
        if os.path.exists(f'{config.net_weights}/ocr_net.pt'):
            self.net.load_state_dict(torch.load(f'{config.net_weights}/ocr_net.pt', map_location='cpu'))
            print('加载参数成功')
        else:
            raise RuntimeError('Model parameters are not loaded')
        self.net = self.net.to(self.device).eval()

    def __call__(self, image):
        with torch.no_grad():
            h, w, c = image.shape
            assert c == 3
            image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_AREA)
            image = torch.from_numpy(image).permute(2, 0, 1) / 255
            image = image.unsqueeze(0).to(self.device)
            out = self.net(image).reshape(-1, 68)
            out = torch.argmax(out, dim=1).cpu().numpy().tolist()
            c = ''
            for i in out:
                c += config.class_names[i]
            return self.deduplication(c)

    def deduplication(self, c):
        """符号去重"""
        temp = ''
        new = ''
        for i in c:
            if i == temp:
                continue
            else:
                if i == config.blank_char:
                    temp = i
                    continue
                new += i
                temp = i
        return new


if __name__ == '__main__':
    e = Bango()
    for image_name in os.listdir('/media/cq/data/public/hibiki/CCPD2019/data/chepai'):
        image = cv2.imread(f'/media/cq/data/public/hibiki/CCPD2019/data/chepai/{image_name}')
        c = e(image)
        print(image_name, c)
        # cv2.destroyAllWindows()
        cv2.imshow('jk', image)
        cv2.waitKey()
