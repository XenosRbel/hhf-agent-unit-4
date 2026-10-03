"""Local eval runner: fetch /questions, run GaiaAgent, save answers, optionally submit."""

from __future__ import annotations
import argparse
import json
import os
import requests
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from agent import GaiaAgent

API_BASE = os.getenv("GAIA_API_BASE", "https://agents-course-unit4-scoring.hf.space")


def fetch_questions() -> list[dict]:
    r = requests.get(f"{API_BASE}/questions", timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--task-id", default="", help="run single task_id only")
    ap.add_argument("--output", default="answers.json")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--username", default=os.getenv("HF_USERNAME", ""))
    ap.add_argument("--code-link", default=os.getenv("AGENT_CODE_LINK", "local-run"))
    ap.add_argument("--model", default=os.getenv("MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    qs = fetch_questions()
    if args.task_id:
        qs = [q for q in qs if q["task_id"] == args.task_id]
    qs = qs[: args.limit]
    print(f"Running {len(qs)} questions with {args.model}")

    agent = GaiaAgent(model=args.model)
    answers, log = [], []
    for i, q in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] {q['task_id']}: {q['question'][:80]}...")
        try:
            ans = agent(q["question"], q["task_id"], q.get("file_name", ""))
        except Exception as e:
            ans = f"AGENT ERROR: {e}"
            print(f"  error: {e} (need MODEL key in .env?)")
        answers.append({"task_id": q["task_id"], "submitted_answer": ans})
        log.append({"task_id": q["task_id"], "question": q["question"], "answer": ans})
        print(f"  -> {str(ans)[:200]}")

    Path(args.output).write_text(
        json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(answers)} answers to {args.output}")

    if args.submit:
        assert args.username, "need --username or HF_USERNAME"
        r = requests.post(
            f"{API_BASE}/submit",
            json={
                "username": args.username,
                "agent_code": args.code_link,
                "answers": answers,
            },
            timeout=60,
        )
        print(r.status_code, r.text[:1000])
        r.raise_for_status()
        d = r.json()
        print(
            f"Score: {d.get('score')}% ({d.get('correct_count')}/{d.get('total_attempted')})"
        )


if __name__ == "__main__":
    main()
