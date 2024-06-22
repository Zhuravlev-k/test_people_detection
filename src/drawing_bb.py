"""
Модуль для отрисовки баундбоксов

Функции:
- draw_predictions: Функция для отрисовки результатов предсказаний (боксов, меток и оценок) на фрейме

Пример использования:
    image = cv.imread('image.jpg')
    predictions = {
        'boxes': [[...], [...]],
        'labels': [...],
        'scores': [...]
    }
    result_image = draw_predictions(image, predictions)
"""

import cv2 as cv

def draw_predictions(image, predictions, threshold=0.5):
    """
    Отрисовывает результаты предсказаний (боксов, меток и оценок) на фрейме

    Параметры:
    image: Изображение в формате numpy/torch.tensor
    predictions: Словарь с предсказанниями
    threshold (float): Порог уверенности модлели для отрисовки бб (по умолчанию 0.5)

    Возвращает:
    Изображение с отрисованными предсказаниями 
    """
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x_left, y_left, x_right, y_right = map(int, box)
            cv.rectangle(image, (x_left, y_left), (x_right, y_right), (255, 0, 0), 2)
            label_text = f"{label} (Person): {score:.2f}"
            cv.putText(image, label_text, (x_left, y_left-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image