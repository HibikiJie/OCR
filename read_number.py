from kenshutsu.detect import Kenshutsu
from bango import Bango
import numpy
from PIL import Image, ImageDraw, ImageFont


class ReadLCD:

    def __init__(self, is_cuda=False):
        self.kenshutsu = Kenshutsu(is_cuda)
        self.bango = Bango()

    def __call__(self, image):
        boxes = self.kenshutsu(image)
        if boxes.size == 0:
            return []
        boxes_new = []
        for box in boxes:
            x1,y1,x2,y2 = box[:4]
            _w, _h = x2 - x1, y2 - y1
            # x1, y1, x2, y2 = x1 - 0.2 * _h, y1 - 0.1 * _h, x2 + 0.2 * _h, y2 + 0.1 * _h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            image_number = image[y1:y2,x1:x2]
            chars = self.bango(image_number)
            # print(chars)
            boxes_new.append([chars,[int(i) for i in box[:4]]])
        return boxes_new


def DrawChinese(img, text, positive, fontSize=20, fontColor=(0, 255, 0)):  # args-(img:numpy.ndarray, text:中文文本, positive:位置, fontSize:字体大小默认20, fontColor:字体颜色默认绿色)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("Chinese.TTC", fontSize, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text(positive, text, fontColor, font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体格式
    cv2charimg = cv2.cvtColor(numpy.array(pilimg), cv2.COLOR_RGB2BGR)  # PIL图片转cv2 图片

    return cv2charimg

if __name__ == '__main__':
    import os
    import cv2
    import re
    c = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
         "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W',
                 'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
           'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    root = '/home/cq/pubilic/hibiki/CCPD2019/ccpd_weather'
    R = ReadLCD(True)
    i = 0
    correct = 0
    for image_name in os.listdir(root):
        try:
            image = cv2.imread(f'{root}/{image_name}')

            target = re.split('[_-]', image_name)
            xy1 = target[3].split('&')
            x1, y1 = xy1
            xy2 = target[4].split('&')
            x2, y2 = xy2
            chepai = target[9:16]
            text = c[int(chepai[0])] + alphabets[int(chepai[1])] + ads[int(chepai[2])] + ads[int(chepai[3])] + ads[
                int(chepai[4])] + ads[int(chepai[5])] + ads[int(chepai[6])]

            boxes = R(image)
            if len(boxes) == 0:
                pass
            else:
                for box in boxes:
                    char,[x1, y1, x2, y2] = box
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    image = DrawChinese(image,char,(x1,y1-50),40,(255,0,0))
                    if char == text:
                        correct+=1
            cv2.imshow('a', image)
            i+=1
            if cv2.waitKey(0) == ord('c'):
                pass
        except:
            pass
    print(i,correct)
    print(correct/i)