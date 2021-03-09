from bango import Bango
import os
import cv2


class Tester:

    def __init__(self):
        super(Tester, self).__init__()
        self.bango = Bango()

    def test(self, root):
        count = 0
        correct = 0
        for image_name in os.listdir(root):
            image_path = f'{root}/{image_name}'
            target = image_name[:-4].split('_')[-1]
            image = cv2.imread(image_path)

            predict = self.bango(image)
            if predict == target:
                correct += 1
            count += 1
            # print(correct,count)
        print(correct, count, '准确率：', correct / count)


if __name__ == '__main__':
    tester = Tester()
    tester.test(root='/media/cq/data/public/hibiki/CCPD2019/data/chepai')
