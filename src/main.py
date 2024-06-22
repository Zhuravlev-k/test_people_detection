"""
Главный модуль для детекции людей на видео и отрисовки предсказаний 

Функции:
- main: функция для обработки видеофайла, выполнения детекции и сохранения видео

Пример использования:
    python main.py
"""

import cv2 as cv
from detection import PeopleDetector
from drawing_bb import draw_predictions
import os

SOURCE_PATH = 'data/crowd.mp4'
THRESHOLD = 0.9

def main(SOURCE_PATH: str = SOURCE_PATH, SAVE_PATH: str = "output/sample_output.avi") -> None:
    """
    Обрабатывает видеофайл, выполняет детекцию людей на каждом кадре и сохраняет обработанное видео

    Параметры:
    SOURCE_PATH (str): Путь к исходному видеофайлу.
    SAVE_PATH (str): Путь для сохранения обработанного видеофайла.

    Возвращает:
    None
    """
    # захватываем видео с источника
    cap = cv.VideoCapture(SOURCE_PATH)
    if not cap.isOpened():
        print("Источник видео не открывается")
        return 

    # инициализируем класс детектора
    detector = PeopleDetector()

    # получаем параметры видео
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    # создаем объект для записи видео
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # Используем кодек XVID просто потому что другой у меня не поехал
    out = cv.VideoWriter(SAVE_PATH, fourcc, fps, (width, height))
    print("Процесс пошёл... Ожидайте")
    while True:
        # распихиваем флаг потока и фреймы по переменным
        ret, frame = cap.read()

        # завершаем цикл если флаг опущен
        if not ret:
            print("Кадры закончились")
            break

        # детектируем моделью
        preds = detector.detect(frame)

        # отрисовываем на кадре предсы
        frame = draw_predictions(frame, preds, threshold=THRESHOLD)

        # записываем результат
        out.write(frame)


        # освобождаем ресурсы
    cap.release()
    out.release()
    # cv.destroyAllWindows() опять же не работает на винде. По идее, на unix надо расскоментить
    print("Результат сохранён")

    # проверка был ли файл сохранен
    if not os.path.exists(SAVE_PATH):
        print("Ошибка. Видео не сохранилдось")

if __name__ == "__main__":
    main()