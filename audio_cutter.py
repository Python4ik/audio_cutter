import os
import shutil
import librosa
import numpy as np
from tqdm import tqdm

# Определение порогов для категорий
BPM_THRESHOLDS = {
    'dance': (120, float('inf')),      # Танцевальная: >120 BPM
    'calm': (60, 120),                  # Спокойная: 60-120 BPM
    'medium': (40, 60)                  # Среднего темпа: 40-60 BPM
}

DYNAMIC_RANGE_THRESHOLD = 0.8  # Динамический диапазон для танцевальных треков

def analyze_track(file_path):
    """Анализирует трек и возвращает его характеристики."""
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Анализ BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo[0]  # Извлекаем скалярное значение из массива

        # Анализ динамического диапазона
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = librosa.amplitude_to_db(np.max(rms) - np.min(rms), ref=np.max)

        # Анализ частотного спектра (средний центр частоты)
        stft = np.abs(librosa.stft(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr))

        return {
            'bpm': tempo,
            'dynamic_range': dynamic_range,
            'spectral_centroid': spectral_centroid
        }
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

def classify_track(track_data):
    """Классифицирует трек на основе его характеристик."""
    bpm = track_data['bpm']
    dynamic_range = track_data['dynamic_range']

    for category, (low, high) in BPM_THRESHOLDS.items():
        if low <= bpm < high:
            if category == 'dance' and dynamic_range < DYNAMIC_RANGE_THRESHOLD:
                continue  # Танцевальные треки должны иметь достаточный динамический диапазон
            return category
    return 'unknown'

def process_audio_files(base_dir):
    """Проходит по всем аудиофайлам в директории и поддиректориях, анализирует их и перемещает."""
    for root, _, files in os.walk(base_dir):
        for file in tqdm(files, desc=f"Обработка директории {root}"):
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):  # Поддерживаемые форматы
                audio_file = os.path.join(root, file)
                track_data = analyze_track(audio_file)

                if track_data is not None:
                    bpm = track_data['bpm']
                    dynamic_range = track_data['dynamic_range']
                    spectral_centroid = track_data['spectral_centroid']

                    category = classify_track(track_data)
                    destination_dir = os.path.join(root, category)

                    # Создание директории для категории, если её нет
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)

                    # Перемещение файла в соответствующую категорию
                    destination_file = os.path.join(destination_dir, file)
                    shutil.move(audio_file, destination_file)

                    # Логирование для диагностики
                    print(f"Файл {file} ({bpm:.2f} BPM, Dynamic Range: {dynamic_range:.2f}, Spectral Centroid: {spectral_centroid:.2f}) перемещён в категорию '{category}'.")
                else:
                    print(f"Не удалось проанализировать файл {file}")

if __name__ == "__main__":
    audio_src = input("Введите путь к исходной директории с аудиофайлами: ").strip()

    if os.path.exists(audio_src):
        process_audio_files(audio_src)
        print("Обработка завершена.")
    else:
        print("Указанный путь не существует.")
