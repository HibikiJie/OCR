import argparse
import numpy
import cv2
import torch
import sys
from kenshutsu.models.experimental import attempt_load
from kenshutsu.utils.general import non_max_suppression,set_logging
from kenshutsu.utils.torch_utils import select_device
sys.path.append('./kenshutsu')


class Kenshutsu:
    def __init__(self, is_cuda=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='/media/cq/data/public/hibiki/OCRNetwork/kenshutsu/runs/train/exp9/weights/best.pt',
                            help='model.pt path(s)')
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        device = '1' if is_cuda else 'cpu'
        parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.opt = parser.parse_args()
        with torch.no_grad():
            weights = self.opt.weights
            set_logging()
            self.device = select_device(self.opt.device)
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model

    def __call__(self, image):
        # image_o = image.copy()
        h1, w1, c = image.shape
        max_len = max(h1, w1)
        fx = 416 / max_len
        fy = 416 / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, c2 = image.shape
        background = numpy.zeros((640, 640, 3), dtype=numpy.uint8)
        background[0:h2, 0:w2] = image
        image = background
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = numpy.transpose(image, axes=(2, 0, 1)) / 255
        image = torch.from_numpy(image).float().to(self.device)
        image = image.unsqueeze(0)
        pred = self.model(image, augment=self.opt.augment)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                   agnostic=self.opt.agnostic_nms)
        pred = pred[0].cpu().numpy()
        pred[:, 0] = (pred[:, 0] / w2) * w1
        pred[:, 2] = (pred[:, 2] / w2) * w1
        pred[:, 1] = (pred[:, 1] / h2) * h1
        pred[:, 3] = (pred[:, 3] / h2) * h1
        return pred


if __name__ == '__main__':
    import os

    e = Kenshutsu(True)
    for image_name in os.listdir('/home/cq/public/hibiki/CCPD2019/ccpd_base'):
        image = cv2.imread(f'/home/cq/public/hibiki/CCPD2019/ccpd_base/{image_name}')
        boxes = e(image)
        if boxes.size == 0:
            pass
        else:
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.imshow('a', image)
        if cv2.waitKey() == ord('c'):
            pass
