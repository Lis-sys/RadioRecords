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

# --- Конфигурация --- #
SAMPLERATE = 16000  # Гц
CHANNELS = 1
BLOCKSIZE = 8000  # 0.5 секунды аудио

MODEL_PATH = "vosk-model-ru-0.42"
LOG_FILE = "transcriptions_stream.log"

EXPECTED_PHRASES = [
    "приём", "как слышно", "конец связи", "северо-восток", "юго-запад",
    "атакую", "поддержка", "узел связи", "пеликан", "фокус-покус", "кубань",
    "[unk]"
]

# --- Функции --- #

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

    wav_dir = "wav_records"
    os.makedirs(wav_dir, exist_ok=True)

    print("\nЗапуск аудиопотока в потоковом режиме...")
    print("Слушаю эфир... Нажмите Ctrl+C для остановки.")

    # Буфер для записи блоков аудио с речью
    speech_audio_buffer = []

    try:
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, blocksize=BLOCKSIZE, device=input_device_index,
                            dtype='int16') as stream:
            while True:
                data, overflowed = stream.read(BLOCKSIZE)
                if overflowed:
                    print("Переполнение аудио буфера!", file=sys.stderr)

                # Преобразуем для подачи в vosk (шумоподавление делаем после буфера, чтобы он не ломал исходные данные)
                audio_float = data.flatten().astype(np.float32) / 32768.0
                reduced_noise_audio = nr.reduce_noise(y=audio_float, sr=SAMPLERATE, stationary=True, prop_decrease=0.8)
                audio_bytes = (reduced_noise_audio * 32768.0).astype(np.int16).tobytes()

                if rec.AcceptWaveform(audio_bytes):
                    # Фраза завершена - достаем текст
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        log_and_print(text, log_path)

                    # Сохраняем накопленные блоки речи в один WAV файл
                    if speech_audio_buffer:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_filename = os.path.join(wav_dir, f"speech_{timestamp}.wav")
                        # объединяем буфер (список numpy массивов int16)
                        full_audio = np.concatenate(speech_audio_buffer).tobytes()
                        save_wav(wav_filename, full_audio, SAMPLERATE, CHANNELS)
                        speech_audio_buffer = []  # очищаем буфер после записи

                else:
                    # Идет речь, собираем блоки в буфер
                    partial_result = json.loads(rec.PartialResult())
                    partial_text = partial_result.get('partial', '')
                    if partial_text:
                        print(f"Распознается: {partial_text}", end='\r')
                        # Сохраняем оригинальный необработанный блок данных в буфер для записи при окончании фразы
                        speech_audio_buffer.append(data.copy().flatten())

    except KeyboardInterrupt:
        print("\nОстановка программы...")
        final_result = json.loads(rec.FinalResult())
        final_text = final_result.get('text', '')
        if final_text:
            print("Последняя распознанная фраза:")
            log_and_print(final_text, log_path)

        # При выходе сохраняем текущий буфер, если есть
        if speech_audio_buffer:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_filename = os.path.join(wav_dir, f"speech_{timestamp}.wav")
            full_audio = np.concatenate(speech_audio_buffer).tobytes()
            save_wav(wav_filename, full_audio, SAMPLERATE, CHANNELS)

    except Exception as e:
        print(f"\nПроизошла критическая ошибка: {e}", file=sys.stderr)
    finally:
        print("Программа завершена.")


if __name__ == "__main__":
    main()
