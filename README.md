---
title: hhf-agent-unit-4
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
hf_oauth: true
---

# hhf-agent-unit-4 — GAIA Level-1 агент

ReAct-агент на LiteLLM + DuckDuckGo. Текущий скор: **55% (11/20)**, порог сертификата — 30%.

## Запуск

```bash
python3 -m pip install --break-system-packages -r requirements.txt
cp .env.example .env   # заполнить MODEL, ключи, HF_TOKEN
```

Переменные `.env`:

| Переменная | Зачем |
|---|---|
| `MODEL` | Reasoning-модель (`openai/...` для router; голое имя само получит префикс `openai/`) |
| `MEDIA_MODEL` | Мультимодальная модель для `analyze_image` / `transcribe_audio` (по умолчанию = `MODEL`) |
| `LITELLM_API_BASE` / `OPENAI_API_BASE` | Свой LiteLLM router (OpenAI-compatible) |
| `LITELLM_API_KEY` / `OPENAI_API_KEY` | Ключ роутера |
| `HF_TOKEN` | Скачивание файлов GAIA (датасет gated, принять условия на странице `gaia-benchmark/GAIA`) |
| `HF_USERNAME` / `AGENT_CODE_LINK` | Для сабмита |
| `MAX_STEPS` / `DEBUG` | Лимит шагов (12), дебаг tool-вызовов (`DEBUG=1`) |

Проверки без ключа (детерминированные ответы):

```bash
python3 eval.py --task-id 2d83110e-a098-4ebb-9987-066c06fa42d0 --output /tmp/a1.json  # -> right
python3 eval.py --task-id 6f37996b-2ac7-44b0-8e68-6d28256631b4 --output /tmp/a2.json  # -> b, e
```

Прогон:

```bash
python3 eval.py --limit 3 --output answers.json    # проба
python3 eval.py --limit 20 --output answers.json   # полный
DEBUG=1 MAX_STEPS=15 python3 eval.py --task-id <id> --output /tmp/x.json  # дебаг одного вопроса
```

## Публикация ответов (бенчмарк)

Скоринг — EXACT MATCH через API `https://agents-course-unit4-scoring.hf.space`:

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv()
import json, os, requests
ans = json.load(open('answers.json'))  # формат: [{\"task_id\": ..., \"submitted_answer\": ...}]
r = requests.post('https://agents-course-unit4-scoring.hf.space/submit',
    json={'username': os.getenv('HF_USERNAME'),
          'agent_code': os.getenv('AGENT_CODE_LINK'),
          'answers': ans}, timeout=60)
print(r.status_code, r.text[:500])
"
```

Требования: `agent_code` — минимум 10 символов (ссылка на код, лучше публичная — Space или GitHub).
Лидерборд хранит **лучший** результат: запись обновляется только если новый скор выше.
Таблица: https://huggingface.co/spaces/agents-course/Students_leaderboard

## Деплой на HF Space (для верификации кода)

Код уже Space-совместим (`app.py` — Gradio, кнопка Run + Submit):

```bash
# один раз: создать public Gradio Space XenosRbel/hhf-agent-unit-4 на huggingface.co/new-space
git remote add space https://huggingface.co/spaces/XenosRbel/hhf-agent-unit-4
git push space main   # нужен HF-токен с Write: hf auth login
```

В Space → Settings → Secrets добавить: `MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`,
`MEDIA_MODEL`, `HF_TOKEN`. Дальше — Login → Run Evaluation & Submit.

## Нюансы

- Эндпоинт `/files/{task_id}` скоринга отвечает 404 на все файлы — `get_task_file` качает
  напрямую из gated-репо `gaia-benchmark/GAIA` через `hf_hub_download` (нужен `HF_TOKEN`).
- `downloads/` (файлы GAIA) в git не коммитятся — решеринг запрещён условиями датасета.
- YouTube без субтитров (`youtube_transcript` пуст) и шахматная тактика через VLM —
  известные слабые места; дешёвые баллы — текст, таблицы, `.py`, `.xlsx`.
