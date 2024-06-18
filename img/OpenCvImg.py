#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2024/6/18 下午11:09
# @Author     : fany
# @Project    : PyCharm
# @File       : OpenCvImg.py
# @Description:
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 加载预训练的 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 视频捕获
cap = cv2.VideoCapture(0)  # 使用摄像头


# 图像预处理
def preprocess_image(frame):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(frame).unsqueeze(0)


# 实时视频目标检测
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 图像预处理
    image = preprocess_image(frame)

    # 进行目标检测
    with torch.no_grad():
        predictions = model(image)

    # 解析预测结果
    boxes = predictions[0]['boxes'].numpy()
    scores = predictions[0]['scores'].numpy()
    labels = predictions[0]['labels'].numpy()

    # 绘制检测结果
    for box, score, label in zip(boxes, scores, labels):
        if score >= 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示视频
    cv2.imshow('Real-time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
