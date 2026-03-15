# stt_bench

Минимальный стенд для сравнения 4 STT backend-ов на macOS/Linux/Windows:

- local (`faster-whisper`)
- OpenAI Realtime STT
- Yandex SpeechKit STT v3 streaming
- ElevenLabs Realtime STT

## Что умеет

- захват аудио с микрофона
- единый формат событий `partial/final/error`
- отдельные demo entrypoint-и для каждого backend-а
- общий `run_compare.py` для сравнения по одному и тому же reference text
- режим сравнения через один и тот же WAV-файл

## Структура

```text
stt_bench/
  requirements.txt
  .env.example
  README.md
  run_local_demo.py
  run_openai_demo.py
  run_yandex_demo.py
  run_elevenlabs_demo.py
  run_compare.py

  common/
  backends/
  scripts/
  fixtures/
```

## Установка

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Все entrypoint-скрипты также умеют сами создать `.venv` и поставить `requirements.txt`, если запускать их вне виртуального окружения и `uv` уже установлен.

## Посмотреть микрофоны

```bash
python scripts/list_audio_devices.py
```

## Локальный backend

```bash
python run_local_demo.py --device default --duration 7
```

## OpenAI

```bash
export OPENAI_API_KEY="..."
python run_openai_demo.py --device default --duration 7
```

## Yandex

```bash
export YANDEX_API_KEY="..."
export YANDEX_STT_ENDPOINT="..."
python run_yandex_demo.py --device default --duration 7 --partial-results
```

## ElevenLabs

```bash
export ELEVENLABS_API_KEY="..."
python run_elevenlabs_demo.py --device default --duration 7
```

## Сравнение

Через микрофон:

```bash
python run_compare.py \
  --reference-text "Привет, это тест распознавания русской речи для сравнения четырёх систем." \
  --duration 7
```

Через один и тот же WAV:

```bash
python run_compare.py \
  --reference-file fixtures/reference_ru_01.txt \
  --wav-file fixtures/reference_ru_01.wav
```

## Важное про Yandex stubs

Для `backends/yandex_backend.py` нужны заранее сгенерированные gRPC stubs из `yandex-cloud/cloudapi`:

Ожидаемые импорты:

- `yandex.cloud.ai.stt.v3.stt_pb2`
- `yandex.cloud.ai.stt.v3.stt_service_pb2_grpc`

## Результаты

Каждый прогон пишет файлы в `results/<run_id>/`:

- `events.jsonl`
- `final_text.txt`
- `compare_summary.json` (для `run_compare.py`)

## Замечания

- Для реально честного сравнения лучше использовать режим `--wav-file`.
- На macOS терминалу/IDE нужно дать доступ к микрофону.
- OpenAI backend автоматически ресемплит вход в 24kHz PCM16 mono.
- ElevenLabs backend лучше всего чувствует себя на `pcm_16000`.
