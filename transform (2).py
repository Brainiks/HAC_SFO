import numpy as np
import os
import pandas as pd
import imageio.v2 as imageio

# Путь к конкретному CSV-файлу
csv_file_path = r'C:\Users\user\PycharmProjects\pythonProject\binary_mask_ndvi (3).csv'

# Проверка существования файла
if os.path.exists(csv_file_path):
    try:
        # Чтение бинарной маски из CSV
        mask = pd.read_csv(csv_file_path, header=None).to_numpy()

        # Сохранение маски в TIFF
        tiff_file_name = os.path.basename(csv_file_path).replace('.csv', '.tiff')
        tiff_path = os.path.join(os.path.dirname(csv_file_path), tiff_file_name)
        imageio.imwrite(tiff_path, mask.astype(np.uint8) * 255)  # Преобразование в 0-255

        print(f"Файл сохранен: {tiff_path}")

    except Exception as e:
        print(f"Ошибка при обработке файла {csv_file_path}: {e}")
else:
    print(f"Файл не найден: {csv_file_path}")
