from typing import List

import cv2
import numpy as np
import torch
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.torch_utils import select_device


class BoundingBox:
    def __init__(self, class_id, label, confidence, bbox, image_width, image_height):
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # t,l,b,r or x1,y1,x2,y2
        self.bbox_normalized = np.array(bbox) / (image_width, image_height, image_width, image_height)
        self.__x1 = bbox[0]
        self.__y1 = bbox[1]
        self.__x2 = bbox[2]
        self.__y2 = bbox[3]
        self.__u1 = self.bbox_normalized[0]
        self.__v1 = self.bbox_normalized[1]
        self.__u2 = self.bbox_normalized[2]
        self.__v2 = self.bbox_normalized[3]

    @property
    def width(self):
        return self.bbox[2] - self.__x1

    @property
    def height(self):
        return self.__y2 - self.__y1

    @property
    def center_absolute(self):
        return 0.5 * (self.__x1 + self.__x2), 0.5 * (self.__y1 + self.__y2)

    @property
    def center_normalized(self):
        return 0.5 * (self.__u1 + self.__u2), 0.5 * (self.__v1 + self.__v2)

    @property
    def size_absolute(self):
        return self.__x2 - self.__x1, self.__y2 - self.__y1

    @property
    def size_normalized(self):
        return self.__u2 - self.__u1, self.__v2 - self.__v1

    def __repr__(self) -> str:
        return f'BoundingBox(class_id: {self.class_id}, label: {self.label}, bbox: {self.bbox}, confidence: {self.confidence:.2f})'


def _preprocess(img, input_shape, letter_box=True, half=False, device=None):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    if device:
        img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def _postprocess(boxes, scores, classes, labels, img_w, img_h):
    if len(boxes) == 0:
        return boxes

    detected_objects = []
    for box, score, class_id, label in zip(boxes, scores, classes, labels):
        detected_objects.append(BoundingBox(class_id, label, score, box, img_w, img_h))
    return detected_objects


class YoloV7Detector:
    def __init__(self, model_name="yolov7.pt", img_size=640, device=''):
        self.device = select_device(device=device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(model_name, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.labels]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        self._id2labels = {i: label for i, label in enumerate(self.labels)}
        self._labels2ids = {label: i for i, label in enumerate(self.labels)}

    @torch.no_grad()
    def detect(self, image, thresh=0.25, iou_thresh=0.45, classes=None, class_labels=None, agnostic=True):
        """:
        """
        img = _preprocess(image, input_shape=(self.img_size, self.img_size), letter_box=True, half=self.half,
                          device=self.device)
        pred = self.model(img, augment=False)[0]
        if not classes and class_labels:
            classes = self.labels2ids(class_labels)
        pred = non_max_suppression(pred, thresh, iou_thres=iou_thresh, classes=classes, agnostic=agnostic)
        det = pred[0]
        boxes = []
        confidences = []
        class_ids = []
        if len(det) > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                t, l, b, r = np.array(xyxy).astype(int)
                boxes.append([t, l, b, r])
                confidences.append(float(conf))
                class_ids.append(int(cls))
        else:
            return []
        labels = [self._id2labels[class_id] for class_id in class_ids]
        detections = _postprocess(boxes, confidences, class_ids, labels, image.shape[1], image.shape[0])
        return detections

    def labels2ids(self, labels: List[str]):
        return [self._labels2ids[label] for label in labels]
