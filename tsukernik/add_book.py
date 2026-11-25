#!/usr/bin/env python3
"""
Универсальный многопоточный импортёр новых шашечных книг (.pdn или .json) в базу приложения.

Примеры:
  python add_book.py --input my_book.pdn --section "Б. Блиндер — Избранные комбинации" --prefix blinder
  python add_book.py --input diagrams.json --section "Новый сборник" --prefix newbook --autosolve
  python add_book.py --rebuild-only

Что делает:
1. Загружает .pdn (через pdn_builder.load_pdn_file) или .json (список позиций / задач).
2. Если у позиции нет mainLine (или включён --autosolve), запускает многоядерный
   TacticalDraughtsSolver (depth=12) для автопоиска и проверки единственности комбинации.
3. Проверяет задачи через audit_puzzle и сохраняет чистый сборник в books/<prefix>.json.
4. Запускает enrich_and_build.py, который пересобирает tsukernik_gold.json и puzzles_data.js
   с калибровкой сложности (1..4), тегами и предвычислением altWins.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audit_master import audit_puzzle
from draughts_solver import TacticalDraughtsSolver, evaluate_position
from pdn_builder import load_pdn_file
from tsukernik_validator import RussianDraughtsEngine

BOOKS_DIR = ROOT / "books"


def _solve_raw_item(args_tuple) -> Optional[dict]:
    item, section, prefix, idx = args_tuple
    ml = item.get("mainLine", [])
    if ml and len(ml) >= 3:
        cand = {
            "id": item.get("id") or f"{prefix}-{item.get('number', idx)}",
            "number": item.get("number", idx),
            "section": section,
            "difficulty": item.get("difficulty", 2),
            "author": item.get("author", ""),
            "goal": item.get("goal", "win"),
            "white": sorted(item.get("white", [])),
            "black": sorted(item.get("black", [])),
            "whiteKings": sorted(item.get("whiteKings", [])),
            "blackKings": sorted(item.get("blackKings", [])),
            "turn": item.get("turn", "w"),
            "mainLine": ml,
            "branches": item.get("branches", []),
            "sources": item.get("sources") or [section],
        }
        return cand if audit_puzzle(cand)[0] == "gold" else None

    # Позиция без готового решения — автопоиск через TacticalDraughtsSolver
    w = item.get("white", [])
    b = item.get("black", [])
    wk = item.get("whiteKings", [])
    bk = item.get("blackKings", [])
    if len(w) + len(wk) < 2 or len(b) + len(bk) < 2:
        return None
    eng = RussianDraughtsEngine(w, b, wk, bk, turn=item.get("turn", "w"))
    solver = TacticalDraughtsSolver()
    res = solver.find_combination(eng, max_depth=12, time_limit_sec=1.6, min_advantage=180, min_gap_over_second=100)
    if not res.get("found"):
        return None
    pv = res.get("pv", [])
    if len(pv) < 3:
        return None

    cur = eng.clone()
    line: List[str] = []
    for i, m_str in enumerate(pv[:14]):
        mv = next((m for m in cur.legal_moves() if m.to_full_notation() == m_str), None)
        if mv is None:
            break
        cur.apply(mv)
        line.append(m_str)
        if cur.turn != item.get("turn", "w") and i >= 2 and not any(m.is_capture for m in cur.legal_moves()):
            sc_opp = -evaluate_position(cur)
            if sc_opp >= 180:
                break
    if len(line) < 3 or cur.turn == item.get("turn", "w") or any(m.is_capture for m in cur.legal_moves()):
        return None

    cand = {
        "id": item.get("id") or f"{prefix}-{item.get('number', idx)}",
        "number": item.get("number", idx),
        "section": section,
        "difficulty": 2,
        "author": item.get("author", ""),
        "goal": "win",
        "white": sorted(w),
        "black": sorted(b),
        "whiteKings": sorted(wk),
        "blackKings": sorted(bk),
        "turn": item.get("turn", "w"),
        "mainLine": line,
        "branches": [],
        "sources": [section],
    }
    return cand if audit_puzzle(cand)[0] == "gold" else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Импорт новой шашечной книги (.pdn / .json) в базу приложения")
    parser.add_argument("--input", type=Path, help="Путь к .pdn или .json файлу книги")
    parser.add_argument("--section", default="Дополнительный сборник", help="Название раздела / книги в приложении")
    parser.add_argument("--prefix", default="book", help="Префикс ID (латиницей), напр. blinder")
    parser.add_argument("--rebuild-only", action="store_true", help="Просто пересобрать puzzles_data.js с текущими книгами")
    args = parser.parse_args()

    if not args.rebuild_only:
        if not args.input or not args.input.exists():
            raise SystemExit(f"Файл не найден: {args.input}")
        if args.input.suffix.lower() == ".pdn":
            raw_items = load_pdn_file(args.input, section=args.section, id_prefix=args.prefix)
        else:
            raw_items = json.loads(args.input.read_text(encoding="utf-8"))

        workers = min(10, max(2, (os.cpu_count() or 4) - 2))
        tasks = [(it, args.section, args.prefix, i + 1) for i, it in enumerate(raw_items)]
        clean: List[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_solve_raw_item, tasks, chunksize=8):
                if r is not None:
                    clean.append(r)

        BOOKS_DIR.mkdir(exist_ok=True)
        out_book = BOOKS_DIR / f"{args.prefix}.json"
        out_book.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(clean)}/{len(raw_items)} verified puzzles -> {out_book}")

    subprocess.run([sys.executable, str(ROOT / "enrich_and_build.py")], check=True)


if __name__ == "__main__":
    main()
