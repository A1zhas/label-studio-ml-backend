import torch
import yaml
from ultralytics import YOLO


def main():
    yaml_file_path = "pivot/data.yaml"
    with open(yaml_file_path, 'r') as file:
        dataset_info = yaml.safe_load(file)

    print("Путь к обучающим данным:", dataset_info['train'])
    print("Путь к валидационным данным:", dataset_info['val'])
    print("Количество классов:", dataset_info['nc'])
    print("Имена классов:", dataset_info['names'])

    # Загрузка предварительно обученной модели (рекомендуется для обучения)
    model = YOLO('yolov8s.pt')

    # Обучение модели
    results = model.train(
        data='pivot/data.yaml',
        epochs=200,
        imgsz=640,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Сохранение обученной модели
    model.export(format='onnx')


if __name__ == '__main__':
    main()