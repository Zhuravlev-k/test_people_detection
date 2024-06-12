"""
Модуль для детекции людей с использованием предобученной Faster R-CNN 

PeopleDetector: Класс для загрузки модели и детекции объектов на изображениях

Пример использования:
    detector = PeopleDetector()
    frame = ...  # код для загрузки изображения
    predictions = detector.detect(frame)
"""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from typing import Dict

class PeopleDetector:
    def __init__(self):
        # Загружаем предобученную Faster R-CNN
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Преобразуем изображение в тензор и переносим на device
        frame_tensor = F.to_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(frame_tensor)

        return predictions[0]