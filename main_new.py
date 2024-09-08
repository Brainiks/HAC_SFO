import numpy as np
import pandas as pd
import os
from glob import glob
import imageio.v2 as imageio
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def read_meteorological_data(folder_path, num_rows=10):
    """
    Функция для чтения метеорологических данных из первого попавшегося CSV.
    Возвращает последние num_rows строк, удаляет столбцы с более 50% пропущенных значений
    и заполняет оставшиеся пропуски средним значением по столбцу.
    """
    try:
        # Ищем первый CSV файл в папке
        csv_files = glob(os.path.join(folder_path, '*.csv'))
        if csv_files:
            csv_file = csv_files[0]  # Берём первый попавшийся CSV
            print(f"Чтение CSV файла: {csv_file}")

            # Читаем CSV файл с указанием разделителя
            df = pd.read_csv(csv_file, sep=',')
            print(f"Заголовки столбцов CSV: {df.columns.tolist()}")  # Выводим заголовки

            # Проверяем наличие столбца 'Дата' или 'time'
            if 'Дата' not in df.columns and 'time' not in df.columns:
                # Пробуем найти столбцы, похожие на 'Дата'
                possible_columns = [col for col in df.columns if 'дата' in col.lower() or 'time' in col.lower()]
                if possible_columns:
                    print(f"Найден альтернативный столбец: {possible_columns[0]}")
                    df.rename(columns={possible_columns[0]: 'Дата'}, inplace=True)
                else:
                    raise KeyError("Столбец 'Дата' или 'time' не найден в CSV-файле")
            elif 'time' in df.columns:
                df.rename(columns={'time': 'Дата'}, inplace=True)

            # Преобразование столбца 'Дата' в формат datetime
            try:
                df['Дата'] = pd.to_datetime(df['Дата'], format='%Y-%m-%d',
                                            errors='coerce')  # Coerce to handle non-parsable dates
                df = df.dropna(subset=['Дата'])  # Удаляем строки с неверными датами
            except Exception as e:
                print(f"Ошибка преобразования столбца 'Дата': {e}")
                return None

            # Удаление столбцов с более чем 50% пропущенных значений
            threshold = len(df) * 0.5
            df = df.dropna(thresh=threshold, axis=1)
            print(f"После удаления столбцов с более 50% пропущенных значений:\n{df.head()}")

            # Заполнение оставшихся пропущенных значений средним по столбцу
            df.fillna(df.mean(numeric_only=True), inplace=True)
            print(f"После заполнения пропущенных значений:\n{df.tail()}")

            # Возвращаем последние строки
            return df.tail(num_rows)
        else:
            print(f"CSV файл не найден в папке: {folder_path}")
            return None
    except Exception as e:
        print(f"Ошибка чтения метеоданных: {e}")
        return None


def prepare_data_with_meteo(folder_path):
    X = []
    y = []

    # Чтение метеоданных
    meteo_data = read_meteorological_data(folder_path)

    for image_path in glob(os.path.join(folder_path, '*.tiff')):
        try:
            image = imageio.imread(image_path)

            if image.ndim < 3 or image.shape[2] != 5:
                print(f"Пропущено изображение (не 5 каналов): {image_path}")
                continue

            input_data = image[..., :4]  # Первые 4 канала
            binary_mask = image[..., 4]  # Пятый канал как бинарная маска

            # Вычисление NDVI, EVI и SAVI
            red_channel = input_data[..., 2]
            nir_channel = input_data[..., 3]
            ndvi = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)
            evi = 2.5 * (nir_channel - red_channel) / (nir_channel + 6 * red_channel - 7.5 * input_data[..., 1] + 1e-10)
            L = 0.5
            savi = (nir_channel - red_channel) / (nir_channel + red_channel + L + 1e-10) * (1 + L)

            ndvi = np.clip(ndvi, -1, 1)
            evi = np.clip(evi, -1, 1)
            savi = np.clip(savi, -1, 1)

            # Добавляем NDVI, EVI и SAVI как новые каналы
            input_data_with_indices = np.dstack((input_data, ndvi, evi, savi))

            # Извлечение даты из имени файла
            timestamp_str = os.path.basename(image_path).split('.')[0]
            timestamp = pd.to_datetime(timestamp_str)
            print(f"Обрабатываем изображение: {image_path}, Дата: {timestamp}")

            # Если метеоданные присутствуют
            if meteo_data is not None and timestamp in meteo_data['Дата'].values:
                meteo_row = meteo_data[meteo_data['Дата'] == timestamp].iloc[0]
                meteo_features = meteo_row[
                    ['Тсред', 'Тмин', 'Тмакс', 'Осадки всего', 'Скорость ветра', 'Атмосферное Давление']].values
            else:
                # Если данных по этой дате нет, используем нулевые значения
                meteo_features = np.zeros(6)

            # Добавляем метеоданные в каждый пиксель
            for i in range(input_data_with_indices.shape[0]):
                for j in range(input_data_with_indices.shape[1]):
                    features = input_data_with_indices[i, j]
                    features_with_meteo = np.concatenate([features, meteo_features])
                    X.append(features_with_meteo)
                    y.append(binary_mask[i, j])

        except Exception as e:
            print(f"Ошибка чтения изображения: {image_path} - {e}")

    return np.array(X), np.array(y)


def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Слой Dropout
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Ещё один слой Dropout
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),  # Нормализация пакета
        layers.Dense(1, activation='sigmoid')  # Для бинарной классификации
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Подготовка данных по всем папкам
all_X = []
all_y = []

base_folder_path = r'C:\Users\user\PycharmProjects\pythonProject\train'

for folder_idx in range(2, 21):  # Для папок '02' до '20'
    folder_path = os.path.join(base_folder_path, f'{folder_idx:02}')  # Формируем путь к папке

    if os.path.exists(folder_path):
        # Подготовка данных с метеоданными
        X, y = prepare_data_with_meteo(folder_path)

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
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

    # Оценка модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Потери: {loss}, Точность: {accuracy}")

    # Сохранение модели
    model.save('trained_model_ndvi_evi_savi_meteo.keras')
    print("Модель сохранена как 'trained_model_ndvi_evi_savi_meteo.keras'")

    # Построение графиков
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
