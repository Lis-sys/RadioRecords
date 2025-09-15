#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np
import noisereduce as nr
from vosk import Model, KaldiRecognizer
import sys
import json
import datetime
import os
import wave
import requests
import re
from scipy.signal import butter, sosfiltfilt  # Для bandpass фильтра

# --- Конфигурация --- #
SAMPLERATE = 16000  # Гц
CHANNELS = 1
BLOCKSIZE = 8000  # 0.5 секунды аудио
MODEL_PATH = "vosk-model-ru-0.42"
LOG_FILE = "transcriptions_stream.log"
wav_dir = "wav_records"

# Конфигурация сервера
SERVER_URL = "https://garrison.hackforces.com/api/voice-chat"
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImhhY2tyZiIsInN1YiI6MSwicm9sZSI6ImhhY2tyZiIsImFybXkiOiJvcmdzIiwiaWF0IjoxNzU3ODI2NTE3LCJleHAiOjE3NTc5MTI5MTd9.9DHvjyLVIgQCx9HHy1bSp-PrRWYaYPFxIEEnTT8lov4"
EXPECTED_PHRASES = [
    "приём", "как слышно", "конец связи", "северо-восток", "юго-запад",
    "атакую", "поддержка", "узел связи", "пеликан", "фокус-покус", "кубань",
    "[unk]", "север", "юг", "восток", "запад", "атака", "огонь", "враг", "союзник",
    "позиция", "координаты", "помощь", "отступление", "подкрепление", "один", "два", "три",
    "четыре", "пять", "шесть", "семь", "восемь", "девять", "ноль", "альфа", "браво", "чарли"
    # Расширенный словарь: числа, команды
]

# Параметры препроцессинга
LOWCUT = 300.0  # Нижняя частота для bandpass (голосовой диапазон)
HIGHCUT = 3000.0  # Верхняя частота
NORMALIZE_DB = -3.0  # Целевая пиковая громкость
MAX_GAIN_DB = 20.0  # Максимальное усиление для тихих сигналов
MIN_PEAK = 0.01  # Минимальный пик для применения нормализации (избежать усиления тишины)
ENERGY_THRESHOLD = 0.03  # Порог RMS для детекции тишины (тюнинговать для дальних сигналов)
SILENCE_THRESHOLD = 4  # Кол-во блоков тишины для принудительного завершения фразы (~1.5с)

# Московское время (UTC+3)
MOSCOW_TZ = datetime.timezone(datetime.timedelta(hours=3))


# --- Функции --- #
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Создает bandpass фильтр."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def bandpass_filter(data, lowcut, highcut, fs):
    """Применяет bandpass фильтр к аудио."""
    sos = butter_bandpass(lowcut, highcut, fs)
    return sosfiltfilt(sos, data)


def normalize_audio(audio, target_db=-3.0, max_gain_db=20.0, min_peak=0.01):
    """Нормализует аудио по пику с ограничениями для тихих/дальних сигналов."""
    peak = np.max(np.abs(audio))
    if peak < min_peak:
        return audio  # Не усиливаем тишину или очень слабый шум
    factor = 10 ** (target_db / 20.0) / peak
    max_factor = 10 ** (max_gain_db / 20.0)
    factor = min(factor, max_factor)
    return audio * factor


def save_wav(filename, data, samplerate, channels):
    """Сохраняет данные audio data в файл WAV."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16 бит = 2 байта
        wf.setframerate(samplerate)
        wf.writeframes(data)


def log_and_print(text, log_file_path):
    """Функция для записи в лог и вывода в консоль."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] - {text}"
    print(f"\033[92m{log_entry}\033[K\033[0m")
    try:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except IOError as e:
        print(f"Ошибка записи в файл {log_file_path}: {e}", file=sys.stderr)


def add_punctuation(text):
    """Добавляет знаки препинания к распознанному тексту на основе простых правил."""
    if not text:
        return text

    if text[-1] not in ['.', '!', '?', ',', ':', ';']:
        text += '.'

    punctuation_rules = [
        (r'\bприём\b', 'приём!'),
        (r'\bатакую\b', 'атакую!'),
        (r'\bподдержка\b', 'поддержка!'),
        (r'\bконец связи\b', 'конец связи.'),
        (r'\bкак слышно\b', 'как слышно?'),
    ]

    for pattern, replacement in punctuation_rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    if len(text) > 0:
        text = text[0].upper() + text[1:]

    return text


def get_moscow_time():
    """Возвращает текущее время в московском формате (UTC+3) в ISO формате без Z."""
    moscow_time = datetime.datetime.now()
    # Форматируем время без информации о временной зоне
    return moscow_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # Убираем микросекунды до миллисекунд


def send_to_server(filename, date, channel, text):
    """Отправляет распознанные данные на сервер."""
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json',
        'Authorization': AUTH_TOKEN
    }

    punctuated_text = add_punctuation(text)

    payload = {
        "file": filename,
        "date": date,
        "channel": channel,
        "text": punctuated_text
    }

    # Логируем отправляемые данные для отладки
    log_and_print(f"Отправка на сервер: {payload}", LOG_FILE)

    try:
        response = requests.post(SERVER_URL, headers=headers, json=payload)
        if response.status_code == 200:
            log_and_print(f"Данные успешно отправлены на сервер: {punctuated_text}", LOG_FILE)
            return True
        else:
            log_and_print(f"Ошибка при отправке данных: {response.status_code} - {response.text}", LOG_FILE)
            return False
    except Exception as e:
        log_and_print(f"Исключение при отправке данных: {e}", LOG_FILE)
        return False


# --- Основная программа --- #
def main():
    print("Инициализация...")
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель Vosk не найдена по пути {MODEL_PATH}")
        print("Пожалуйста, скачайте модель с https://alphacephei.com/vosk/models")
        sys.exit(1)
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLERATE, json.dumps(EXPECTED_PHRASES, ensure_ascii=False))
    print("Модель Vosk загружена.")

    try:
        device_info = sd.query_devices(kind='input')
        input_device_index = sd.default.device[0]
        for i, device in enumerate(sd.query_devices()):
            if 'virtual' in device['name'].lower() or 'cable' in device['name'].lower():
                input_device_index = i
                break
        print(f"Используется устройство ввода: {sd.query_devices(input_device_index)['name']}")
    except Exception as e:
        print(f"Ошибка при поиске аудиоустройств: {e}", file=sys.stderr)
        sys.exit(1)

    log_path = os.path.abspath(LOG_FILE)
    print(f"Логи будут записываться в файл: {log_path}")
    os.makedirs(wav_dir, exist_ok=True)

    print("\nЗапуск аудиопотока в потоковом режиме...")
    print("Слушаю эфир... Нажмите Ctrl+C для остановки.")

    speech_audio_buffer = []  # Буфер для оригинального аудио
    silence_blocks = 0

    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=BLOCKSIZE,
                            device=input_device_index, dtype='int16') as stream:
            while True:
                data, overflowed = stream.read(BLOCKSIZE)
                if overflowed:
                    print("Переполнение аудио буфера!", file=sys.stderr)

                # Препроцессинг
                audio_float = data.flatten().astype(np.float32) / 32768.0
                filtered_audio = bandpass_filter(audio_float, LOWCUT, HIGHCUT, SAMPLERATE)
                reduced_noise_audio = nr.reduce_noise(y=filtered_audio, sr=SAMPLERATE, stationary=False,
                                                      prop_decrease=0.75)
                normalized_audio = normalize_audio(reduced_noise_audio, NORMALIZE_DB, MAX_GAIN_DB, MIN_PEAK)

                # Вычисление RMS для детекции тишины
                rms = np.sqrt(np.mean(normalized_audio ** 2))

                audio_bytes = (normalized_audio * 32768.0).astype(np.int16).tobytes()

                if rec.AcceptWaveform(audio_bytes):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        log_and_print(text, log_path)

                        if speech_audio_buffer:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            wav_filename = os.path.join(wav_dir, f"speech_{timestamp}.wav")
                            full_audio = np.concatenate(speech_audio_buffer).tobytes()
                            save_wav(wav_filename, full_audio, SAMPLERATE, CHANNELS)

                            moscow_date = get_moscow_time()  # Используем московское время
                            send_to_server(os.path.basename(wav_filename), moscow_date, 4, text)

                        speech_audio_buffer = []
                    silence_blocks = 0
                else:
                    partial_result = json.loads(rec.PartialResult())
                    partial_text = partial_result.get('partial', '')
                    if partial_text:
                        print(f"Распознается: {partial_text}", end='\r')
                        speech_audio_buffer.append(data.copy().flatten())
                        silence_blocks = 0
                    else:
                        if speech_audio_buffer:  # Если была речь
                            if rms < ENERGY_THRESHOLD:
                                silence_blocks += 1
                            else:
                                silence_blocks = 0

                            if silence_blocks >= SILENCE_THRESHOLD:
                                # Принудительное завершение фразы
                                final_result = json.loads(rec.FinalResult())
                                text = final_result.get('text', '')
                                if text:
                                    log_and_print(text, log_path)

                                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    wav_filename = os.path.join(wav_dir, f"speech_{timestamp}.wav")
                                    full_audio = np.concatenate(speech_audio_buffer).tobytes()
                                    save_wav(wav_filename, full_audio, SAMPLERATE, CHANNELS)

                                    moscow_date = get_moscow_time()  # Используем московское время
                                    send_to_server(os.path.basename(wav_filename), moscow_date, 4, text)

                                speech_audio_buffer = []
                                silence_blocks = 0

    except KeyboardInterrupt:
        print("\nОстановка программы...")
        final_result = json.loads(rec.FinalResult())
        final_text = final_result.get('text', '')
        if final_text:
            print("Последняя распознанная фраза:")
            log_and_print(final_text, log_path)

        if speech_audio_buffer:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_filename = os.path.join(wav_dir, f"speech_{timestamp}.wav")
            full_audio = np.concatenate(speech_audio_buffer).tobytes()
            save_wav(wav_filename, full_audio, SAMPLERATE, CHANNELS)

            moscow_date = get_moscow_time()  # Используем московское время
            send_to_server(os.path.basename(wav_filename), moscow_date, 4, final_text)

    except Exception as e:
        print(f"\nПроизошла критическая ошибка: {e}", file=sys.stderr)
    finally:
        print("Программа завершена.")


if __name__ == "__main__":
    main()
