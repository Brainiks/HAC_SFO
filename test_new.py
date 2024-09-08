import numpy as np
import imageio.v2 as imageio
from tensorflow.keras.models import load_model
import pandas as pd
import gradio as gr

def prepare_test_data(image_path):
    X = []
    height, width = None, None
    error = None

    try:
        image = imageio.imread(image_path)

        if image.ndim < 3 or image.shape[2] < 4:
            error = f"Пропущено изображение (не 4 канала): {image_path}"
            return None, None, None, error

        input_data = image[..., :4]
        height, width = input_data.shape[:2]

        red_channel = input_data[..., 2]
        nir_channel = input_data[..., 3]
        blue_channel = input_data[..., 0]


        # NDVI
        ndvi = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)
        ndvi = np.clip(ndvi, -1, 1)

        # EVI
        evi = 2.5 * (nir_channel - red_channel) / (nir_channel + 6 * red_channel - 7.5 * blue_channel + 1e-10)
        evi = np.clip(evi, -1, 1)

        # SAVI
        L = 0.5
        savi = (nir_channel - red_channel) / (nir_channel + red_channel + L + 1e-10)
        savi = np.clip(savi, -1, 1)

        for i in range(height):
            for j in range(width):
                features = np.concatenate((input_data[i, j], [ndvi[i, j], evi[i, j], savi[i, j]]))
                X.append(features)

    except Exception as e:
        error = f"Ошибка чтения изображения: {image_path} - {e}"

    return np.array(X), height, width, error

def predict(image_path, csv_path):
    X_test, height, width, error = prepare_test_data(image_path)

    if error:
        return error, None

    try:
        df = pd.read_csv(csv_path, sep=';')
        # Удаляем столбцы с более чем 50% пропущенных значений
        threshold = len(df) * 0.5
        df = df.dropna(thresh=threshold, axis=1)

        # Заполняем оставшиеся пропуски средним значением
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Подготовка метео-данных
        meteorological_features = df.values

        # Убедитесь, что у вас 13 признаков
        if X_test.shape[1] < 13:
            # Добавляем недостающие признаки, если необходимо
            # Например, заполняем нулями или другим значением
            missing_features = 13 - X_test.shape[1]
            X_test = np.hstack([X_test, np.zeros((X_test.shape[0], missing_features))])

        if X_test.size > 0:
            predictions = model.predict(X_test)
            binary_predictions = (predictions > 0.5).astype(int)

            mask = binary_predictions.reshape((height, width))
            df_mask = pd.DataFrame(mask)
            output_csv_path = 'binary_mask_ndvi.csv'
            df_mask.to_csv(output_csv_path, index=False, header=False)

            return f"Предсказания сохранены в: {output_csv_path}", output_csv_path

    except Exception as e:
        return f"Ошибка обработки данных: {e}", None

    return "Нет данных для тестирования.", None

# Загрузка модели
model = load_model('trained_model_ndvi_evi_savi_meteo.keras')

# Создание интерфейса Gradio
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.File(label="Загрузите TIFF изображение"),
        gr.File(label="Загрузите CSV файл с метео-данными")
    ],
    outputs=[gr.Textbox(label="Сообщение", interactive=False), gr.File(label="Скачать CSV файл")],
    title="ПРОГНОЗИРОВАНИЕ ПОЖАРОВ",
    allow_flagging=False,
    live=False
)

# Запуск интерфейса
iface.launch(share=True)
