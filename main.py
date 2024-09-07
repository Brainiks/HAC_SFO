import numpy as np
import pandas as pd
import os
from glob import glob
import imageio.v2 as imageio
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # Импортируем matplotlib

def prepare_data(folder_path):
    X = []
    y = []

    for image_path in glob(os.path.join(folder_path, '*.tiff')):
        try:
            image = imageio.imread(image_path)

            if image.ndim < 3 or image.shape[2] != 5:
                print(f"Пропущено изображение (не 5 каналов): {image_path} - Каналов: {image.shape[2] if image.ndim > 2 else 'неизвестно'}")
                continue

            input_data = image[..., :4]  # Первые 4 канала
            binary_mask = image[..., 4]   # Пятый канал как бинарная маска

            # Вычисление NDVI
            red_channel = input_data[..., 2]  # Красный канал
            nir_channel = input_data[..., 3]   # NIR канал

            ndvi = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)  # NDVI
            ndvi = np.clip(ndvi, -1, 1)  # Ограничиваем значения NDVI от -1 до 1

            # Вычисление EVI
            evi = 2.5 * (nir_channel - red_channel) / (nir_channel + 6 * red_channel - 7.5 * input_data[..., 1] + 1e-10)  # EVI
            evi = np.clip(evi, -1, 1)  # Ограничиваем значения EVI от -1 до 1

            # Вычисление SAVI
            L = 0.5  # Константа для SAVI
            savi = (nir_channel - red_channel) / (nir_channel + red_channel + L + 1e-10) * (1 + L)  # SAVI
            savi = np.clip(savi, -1, 1)  # Ограничиваем значения SAVI от -1 до 1

            # Добавляем NDVI, EVI и SAVI как новые каналы в input_data
            input_data_with_indices = np.dstack((input_data, ndvi, evi, savi))

            # Извлечение даты из имени файла
            timestamp_str = os.path.basename(image_path).split('.')[0]
            timestamp = pd.to_datetime(timestamp_str)  # Конвертация в datetime
            print(f"Обрабатываем изображение: {image_path}, Дата: {timestamp}")

            # Формирование данных
            for i in range(input_data_with_indices.shape[0]):
                for j in range(input_data_with_indices.shape[1]):
                    features = input_data_with_indices[i, j]
                    X.append(features)
                    y.append(binary_mask[i, j])

        except Exception as e:
            print(f"Ошибка чтения изображения: {image_path} - {e}")

    return np.array(X), np.array(y)

# Функция для построения модели
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Для бинарной классификации
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Подготовка данных по всем папкам
all_X = []
all_y = []

# Новый путь к папке
base_folder_path = r'C:\Users\user\PycharmProjects\pythonProject\train'

for folder_idx in range(2, 21):  # Для папок '02' до '20'
    folder_path = os.path.join(base_folder_path, f'{folder_idx:02}')  # Формируем путь к папке

    if os.path.exists(folder_path):
        # Подготовка данных из изображений
        X, y = prepare_data(folder_path)

        # Проверка: добавляем только непустые массивы
        if X.size > 0 and y.size > 0:
            all_X.append(X)
            all_y.append(y)

# Объединяем данные из всех папок
if all_X and all_y:
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    # Разделение на обучение и тестирование
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Обучение модели
    model = build_model((X_train.shape[1],))
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Оценка модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Потери: {loss}, Точность: {accuracy}")

    # Сохранение модели в формате Keras
    model.save('trained_model_ndvi_evi_savi.keras')
    print("Модель сохранена как 'trained_model_ndvi_evi_savi.keras'")

    # Построение графиков обучения
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери (обучение)')
    plt.plot(history.history['val_loss'], label='Потери (валидация)')
    plt.title('График потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точность (обучение)')
    plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
    plt.title('График точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("Нет данных для обучения модели.")
