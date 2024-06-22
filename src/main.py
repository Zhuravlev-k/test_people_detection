"""
Главный модуль для детекции людей на видео и отрисовки предсказаний 

Функции:
- main: функция для обработки видеофайла, выполнения детекции и сохранения видео

Пример использования:
    python main.py
"""
import argparse
import cv2 as cv
from detection import PeopleDetector
from drawing_bb import draw_predictions
import os

def process_video(SOURCE_PATH: str , SAVE_PATH: str, threshold: float) -> None:
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
        frame = draw_predictions(frame, preds, threshold=threshold)

        # записываем результат
        out.write(frame)


        # освобождаем ресурсы
    cap.release()
    out.release()
    # cv.destroyAllWindows() опять же не работает на винде. По идее, на unix надо расскоментить
    print("Результат сохранён")

    # проверка был ли файл сохранен
    if not os.path.exists(SAVE_PATH):
        print("Ошибка. Видео не сохранилось")

def main():
    parser = argparse.ArgumentParser(description="Детектирует людей и рисует боксы.")
    parser.add_argument("--input_folder", type=str, default="data", help="Папка с исходным видео.")
    parser.add_argument("--output_folder", type=str, default="output", help="Папка для сохранения обработанного видео.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Трешхолд увернности модели в классе")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in os.listdir(args.input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, f"output_{filename}")
            process_video(input_path, output_path, args.threshold)

if __name__ == "__main__":
    main()