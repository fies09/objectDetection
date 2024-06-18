#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2024/6/18 下午11:08
# @Author     : fany
# @Project    : PyCharm
# @File       : FasterRcnnImg.py
# @Description:
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision

# 加载预训练的 Faster R-CNN 模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 设置为评估模式

# 图像预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 加载并预处理图像
image_path = "path/to/your/image.jpg"
image = preprocess_image(image_path)

# 进行目标检测
with torch.no_grad():
    predictions = model(image)

# 可视化结果
def visualize_predictions(image_path, predictions, threshold=0.5):
    image = cv2.imread(image_path)
    predictions = predictions[0]
    boxes = predictions["boxes"].numpy()
    scores = predictions["scores"].numpy()
    labels = predictions["labels"].numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# 可视化检测结果
visualize_predictions(image_path, predictions)
