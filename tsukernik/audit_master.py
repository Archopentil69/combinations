#!/usr/bin/env python3
"""
Глубокий аудит собранной базы tsukernik_full_master.json:
- строгая проверка всех ходов (главная линия + ветки + altWins) без OCR-поблажек;
- отлов оборванных комбинаций (незавершённый обязательный бой в конце);
- проверка материального/позиционного результата комбинации;
- отделение главы 16 (контрприёмы чёрных).

Запуск:
  python audit_master.py tsukernik_full_master.json
"""

from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tsukernik_validator import RussianDraughtsEngine


def exact_move(eng: RussianDraughtsEngine, move_str: str):
    for mv in eng.legal_moves():
        if mv.to_full_notation() == move_str:
            return mv
    return None


def material_score(eng: RussianDraughtsEngine) -> Tuple[int, int, int, int, int]:
    """Возвращает (score, w, W, b, B), где простая=1, дамка=3."""
    w = sum(1 for p in eng.board.values() if p == "w")
    W = sum(1 for p in eng.board.values() if p == "W")
    b = sum(1 for p in eng.board.values() if p == "b")
    B = sum(1 for p in eng.board.values() if p == "B")
    return (w + 3 * W) - (b + 3 * B), w, W, b, B


def is_chapter_16(puzzle: dict) -> bool:
    num = puzzle.get("number", 0)
    sec = str(puzzle.get("section", "")).lower()
    srcs = " ".join(puzzle.get("sources", [])).lower()
    return (
        "контрприем" in sec
        or "kontrpriem" in srcs
        or (isinstance(num, int) and num >= 2600)
    )


def audit_puzzle(puzzle: dict) -> Tuple[str, Optional[str]]:
    """
    Возвращает (статус, причина):
      - 'gold'    : задача полностью корректна и логична
      - 'ch16'    : глава 16 (выигрывают чёрные, отдельный пул)
      - 'suspect' : подозрительная или битая комбинация
    """
    main_line: List[str] = puzzle.get("mainLine", [])
    if len(main_line) < 2:
        return "suspect", "Слишком короткая линия (< 2 полуходов)"

    eng = RussianDraughtsEngine(
        puzzle.get("white", []),
        puzzle.get("black", []),
        puzzle.get("whiteKings", []),
        puzzle.get("blackKings", []),
        turn=puzzle.get("turn", "w"),
    )
    start_score, _, start_W, _, _ = material_score(eng)
    states: Dict[int, RussianDraughtsEngine] = {0: eng.clone()}

    # 1. Строгая проверка главной линии
    for ply, m_str in enumerate(main_line, start=1):
        mv = exact_move(eng, m_str)
        if mv is None:
            return "suspect", f"Нелегальный ход главной линии на ply {ply}: {m_str}"
        eng.apply(mv)
        states[ply] = eng.clone()

    # 2. Строгая проверка всех веток и альтернативных выигрышей
    for field_name in ("branches", "altWins"):
        for idx, br in enumerate(puzzle.get(field_name, []), start=1):
            after_ply = br.get("afterPly")
            if after_ply not in states:
                return "suspect", f"{field_name}#{idx}: некорректный afterPly={after_ply}"
            br_eng = states[after_ply].clone()
            for j, m_str in enumerate(br.get("moves", []), start=1):
                mv = exact_move(br_eng, m_str)
                if mv is None:
                    return "suspect", f"{field_name}#{idx} ход {j}: нелегальный {m_str}"
                br_eng.apply(mv)

    # Глава 16 (контрприёмы чёрных) — отделяем после проверки легальности
    if is_chapter_16(puzzle):
        return "ch16", None

    # 3. Проверка финальной позиции на обрыв комбинации и смысл
    end_legal = eng.legal_moves()
    end_score, w, W, b, B = material_score(eng)
    is_locked_black = eng.turn == "b" and len(end_legal) == 0
    is_draw = puzzle.get("goal") == "draw"

    # Если ход белых и у белых висит обязательное взятие — линия оборвана
    if eng.turn == "w" and end_legal and end_legal[0].is_capture:
        return "suspect", "Обрыв линии: у белых остался обязательный бой в конце"

    # Если ход чёрных и у них обязательный бой — смотрим результат после ответного боя
    if eng.turn == "b" and end_legal and end_legal[0].is_capture:
        # Оцениваем худший для белых ответный бой чёрных
        worst_after_cap = 999
        for reply in end_legal:
            probe = eng.clone()
            probe.apply(reply)
            sc, _, _, _, _ = material_score(probe)
            worst_after_cap = min(worst_after_cap, sc)
        if not is_draw and worst_after_cap < start_score - 1 and W == 0:
            return "suspect", (
                f"Обрыв перед боем чёрных: материальный баланс падает "
                f"({start_score:+d} -> {worst_after_cap:+d})"
            )

    # Проверка: белые не должны проигрывать материал без дамки и без запирания (кроме ничьих)
    gained_king = W > start_W
    if not is_draw and not is_locked_black and not gained_king:
        if end_score <= start_score - 2 and end_score < 0:
            return "suspect", (
                f"Подозрительный финал: белые потеряли материал "
                f"({start_score:+d} -> {end_score:+d}) без дамки и запирания"
            )

    return "gold", None


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tsukernik_full_master.json")
    data = json.loads(src.read_text(encoding="utf-8"))

    gold: List[dict] = []
    ch16: List[dict] = []
    suspect: List[dict] = []
    reasons: Counter[str] = Counter()

    for p in data:
        status, reason = audit_puzzle(p)
        if status == "gold":
            gold.append(p)
        elif status == "ch16":
            ch16.append(p)
        else:
            p_copy = dict(p)
            p_copy["auditError"] = reason
            suspect.append(p_copy)
            reasons[(reason or "unknown").split(":")[0]] += 1

    Path("tsukernik_gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path("tsukernik_ch16_black.json").write_text(
        json.dumps(ch16, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path("tsukernik_suspect.json").write_text(
        json.dumps(suspect, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print(f"Проверено задач:            {len(data)}")
    print(f"Чистых (tsukernik_gold):    {len(gold)}")
    print(f"Глава 16 (за чёрных):       {len(ch16)}")
    print(f"Подозрительных (отсеяно):   {len(suspect)}")
    if reasons:
        print("\nПричины отсева:")
        for k, v in reasons.most_common():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()