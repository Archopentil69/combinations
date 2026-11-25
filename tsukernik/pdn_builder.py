#!/usr/bin/env python3
"""
Универсальный импортёр PDN + объединитель и дедупликатор базы задач (русские шашки 8x8).
Использует строгий движок из tsukernik_validator.py (должен лежать в той же папке).

Примеры запуска:
  # Конвертировать один PDN-файл:
  python pdn_builder.py build book.pdn --source "В. Плехов" --out plekhov.json

  # Склеить все .pdn и .json из папки в единую базу без дубликатов:
  python pdn_builder.py build ch_a1_a7.json plekhov.pdn adamovich.pdn --out master_puzzles.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tsukernik_validator import (
    FILES,
    Move,
    RussianDraughtsEngine,
    match_move_with_repair,
    rc_to_sq,
)


# ============================================================================
# 1. КОНВЕРТЕР ЦИФРОВОЙ НОТАЦИИ (1–32) <-> АЛГЕБРАИЧЕСКОЙ (a1–h8)
# ============================================================================

def num_to_sq(n: int) -> str:
    """Стандартная нумерация 8x8 PDN: 1=b8, 2=d8, ..., 29=a1, 32=g1."""
    if not (1 <= n <= 32):
        raise ValueError(f"Некорректный номер поля PDN: {n}")
    row_from_top = (n - 1) // 4
    rank = 8 - row_from_top
    col = ((n - 1) % 4) * 2 + (1 if rank % 2 == 0 else 0)
    return f"{FILES[col]}{rank}"


def parse_fen(fen: str) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """
    Парсит FEN в обоих форматах:
      - Цифровой:      "W:W31,32,K27:B1,2,K10"
      - Алгебраический: "W:Wa1,b2,Kd4:Bb4,Kd8"
    """
    fen = fen.strip().strip('"')
    parts = [p.strip() for p in fen.split(":") if p.strip()]
    turn = "w"
    white, black, w_kings, b_kings = [], [], [], []

    for part in parts:
        if part in ("W", "B", "w", "b"):
            turn = part.lower()
            continue
        side = part[0].upper()
        items = [x.strip() for x in part[1:].split(",") if x.strip()]
        for item in items:
            is_king = item.startswith(("K", "k"))
            raw_sq = item[1:] if is_king else item
            if raw_sq.isdigit():
                sq = num_to_sq(int(raw_sq))
            elif re.fullmatch(r"[a-hA-H][1-8]", raw_sq):
                sq = raw_sq.lower()
            else:
                continue
            if side == "W":
                (w_kings if is_king else white).append(sq)
            elif side == "B":
                (b_kings if is_king else black).append(sq)

    return turn, sorted(set(white)), sorted(set(black)), sorted(set(w_kings)), sorted(set(b_kings))


def normalize_pdn_move_token(tok: str) -> str:
    """Переводит числовой ход PDN (22-18, 18x11x4) в алгебраический (c3-d4, c5:e3:g5)."""
    clean = tok.strip("!?+# ")
    # Числовой тихий ход: 22-18
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", clean)
    if m:
        return f"{num_to_sq(int(m.group(1)))}-{num_to_sq(int(m.group(2)))}"
    # Числовое взятие: 18x11 или 18:11:4
    if re.fullmatch(r"\d{1,2}(?:[xX:]\d{1,2})+", clean):
        nums = [int(x) for x in re.split(r"[xX:]", clean)]
        return ":".join(num_to_sq(n) for n in nums)
    # Алгебраический с 'x': e5xc3 -> e5:c3
    return clean.replace("x", ":").replace("X", ":")


def match_pdn_move(tok: str, legal: List[Move]) -> Optional[Move]:
    """Сопоставляет ход из PDN (включая сокращённые взятия start:end) с легальными ходами."""
    norm = normalize_pdn_move_token(tok)
    mv, _ = match_move_with_repair(norm, legal)
    if mv is not None:
        return mv
    # Если в PDN указаны только начало и конец длинного взятия (a1:g7), а в движке полная цепочка
    if ":" in norm:
        parts = norm.split(":")
        if len(parts) == 2:
            s_sq, e_sq = parts
            cands = [m for m in legal if m.is_capture and m.start_sq == s_sq and m.end_sq == e_sq]
            if len(cands) == 1:
                return cands[0]
    return None


# ============================================================================
# 2. ТОКЕНИЗАЦИЯ И РАЗБОР ДЕРЕВА ВАРИАНТОВ PDN
# ============================================================================

PDN_TOKEN_RE = re.compile(
    r"\{[^}]*\}|"            # {комментарий}
    r"\(|\)|"                # скобки ветвлений
    r"1-0|0-1|1/2-1/2|\*|"   # результат партии
    r"\d+\.\.\.|\d+\.|"      # номера ходов 1. или 1...
    r"[^\s(){}]+"            # сам ход
)


def parse_pdn_games(text: str) -> List[Tuple[Dict[str, str], str]]:
    """Разбивает текст PDN-файла на список (теги, текст_ходов)."""
    games: List[Tuple[Dict[str, str], str]] = []
    tags: Dict[str, str] = {}
    moves_lines: List[str] = []
    in_moves = False

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        m_tag = re.match(r'^\[(\w+)\s+"(.*)"\]$', s)
        if m_tag:
            if in_moves and (tags or moves_lines):
                games.append((tags, " ".join(moves_lines)))
                tags, moves_lines, in_moves = {}, [], False
            tags[m_tag.group(1)] = m_tag.group(2)
        else:
            in_moves = True
            moves_lines.append(s)

    if tags or moves_lines:
        games.append((tags, " ".join(moves_lines)))
    return games


def parse_variation_tree(
    tokens: List[str],
    pos: int,
    start_eng: RussianDraughtsEngine,
    base_ply: int,
) -> Tuple[int, List[str], List[dict], List[dict], Optional[str]]:
    """
    Рекурсивно разбирает основную линию и все вложенные скобочные варианты (...).
    Возвращает: (новый_индекс, main_line, branches, alt_wins, error).
    """
    eng = start_eng.clone()
    line_moves: List[str] = []
    branches: List[dict] = []
    alt_wins: List[dict] = []
    state_before_last: Optional[RussianDraughtsEngine] = None
    ply_before_last = base_ply

    i = pos
    while i < len(tokens):
        tok = tokens[i]
        i += 1

        if tok.startswith("{") or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        if re.fullmatch(r"\d+\.(\.\.)?", tok):
            continue
        if tok == ")":
            break
        if tok == "(":
            if state_before_last is None:
                # Пропускаем некорректную скобку в самом начале
                depth = 1
                while i < len(tokens) and depth > 0:
                    if tokens[i] == "(":
                        depth += 1
                    elif tokens[i] == ")":
                        depth -= 1
                    i += 1
                continue

            i, var_moves, sub_br, sub_alt, var_err = parse_variation_tree(
                tokens, i, state_before_last, ply_before_last
            )
            if var_err is None and var_moves:
                if state_before_last.turn == "w":
                    alt_wins.append({"afterPly": ply_before_last, "moves": var_moves})
                else:
                    branches.append({"afterPly": ply_before_last, "moves": var_moves})
                branches.extend(sub_br)
                alt_wins.extend(sub_alt)
            continue

        # Обычный токен хода
        mv = match_pdn_move(tok, eng.legal_moves())
        if mv is None:
            return i, line_moves, branches, alt_wins, f"Нелегальный ход '{tok}' на ply {base_ply + len(line_moves) + 1}"

        state_before_last = eng.clone()
        ply_before_last = base_ply + len(line_moves)
        line_moves.append(mv.to_full_notation())
        eng.apply(mv)

    return i, line_moves, branches, alt_wins, None


# ============================================================================
# 3. ХЕШИРОВАНИЕ, ДЕДУПЛИКАЦИЯ, АВТОТЕГИ И СТАРТОВЫЙ РЕЙТИНГ
# ============================================================================

def position_hash(puzzle: dict) -> str:
    """Уникальный отпечаток позиции (не зависит от книги и номера задачи)."""
    key = "|".join([
        puzzle.get("turn", "w"),
        "W:" + ",".join(sorted(puzzle.get("white", []))),
        "WK:" + ",".join(sorted(puzzle.get("whiteKings", []))),
        "B:" + ",".join(sorted(puzzle.get("black", []))),
        "BK:" + ",".join(sorted(puzzle.get("blackKings", []))),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def compute_tags_and_rating(puzzle: dict) -> Tuple[List[str], int]:
    """Автоматически проставляет тактические теги и стартовый рейтинг (800–2200)."""
    tags = set(puzzle.get("tags", []))
    main_line = puzzle.get("mainLine", [])
    branches = puzzle.get("branches", [])
    alt_wins = puzzle.get("altWins", [])

    if branches:
        tags.add("варианты защиты")
    if alt_wins:
        tags.add("альтернативный выигрыш")
    if puzzle.get("goal") == "draw":
        tags.add("ничья")
    if puzzle.get("whiteKings") or puzzle.get("blackKings"):
        tags.add("дамки")

    # Проверяем, есть ли превращение в дамку посреди боя или тихий первый ход
    eng = RussianDraughtsEngine(
        puzzle.get("white", []),
        puzzle.get("black", []),
        puzzle.get("whiteKings", []),
        puzzle.get("blackKings", []),
        turn=puzzle.get("turn", "w"),
    )
    has_mid_promo = False
    for idx, m_str in enumerate(main_line):
        mv = match_pdn_move(m_str, eng.legal_moves())
        if mv is None:
            break
        if mv.promotes and mv.is_capture and len(mv.path) >= 3:
            has_mid_promo = True
        if idx == 0 and not mv.is_capture:
            tags.add("тихий старт")
        eng.apply(mv)

    if has_mid_promo:
        tags.add("превращение в бою")

    # Стартовый рейтинг по сложности, длине и ветвлениям
    diff = int(puzzle.get("difficulty", 1))
    base = {1: 1000, 2: 1400, 3: 1800}.get(diff, 1200)
    rating = base + min(400, len(main_line) * 35) + (80 if has_mid_promo else 0) + (60 if branches else 0)
    return sorted(tags), min(2200, max(800, rating))


def merge_puzzle(existing: dict, incoming: dict) -> dict:
    """Объединяет дубликат: дополняет источники и уникальные ветки боя/выигрыша."""
    sources = list(existing.get("sources", []))
    for src in incoming.get("sources", []):
        if src and src not in sources:
            sources.append(src)
    existing["sources"] = sources

    # Добавляем новые уникальные ветки защиты
    seen_br = {(b["afterPly"], tuple(b["moves"])) for b in existing.get("branches", [])}
    for br in incoming.get("branches", []):
        sig = (br["afterPly"], tuple(br["moves"]))
        if sig not in seen_br:
            existing.setdefault("branches", []).append(br)
            seen_br.add(sig)

    # Добавляем новые альтернативные выигрыши
    seen_alt = {(a["afterPly"], tuple(a["moves"])) for a in existing.get("altWins", [])}
    for alt in incoming.get("altWins", []):
        sig = (alt["afterPly"], tuple(alt["moves"]))
        if sig not in seen_alt:
            existing.setdefault("altWins", []).append(alt)
            seen_alt.add(sig)

    tags, rating = compute_tags_and_rating(existing)
    existing["tags"] = tags
    existing["rating"] = rating
    return existing


# ============================================================================
# 4. ЗАГРУЗКА PDN И JSON + СБОРКА МАСТЕР-БАЗЫ
# ============================================================================

def load_pdn_file(path: Path, default_source: str) -> Tuple[List[dict], int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    games = parse_pdn_games(text)
    puzzles: List[dict] = []
    failed = 0

    for idx, (tags, moves_text) in enumerate(games, start=1):
        fen = tags.get("FEN")
        if not fen:
            continue
        turn, white, black, w_kings, b_kings = parse_fen(fen)
        if not (white or w_kings) or not (black or b_kings):
            failed += 1
            continue

        eng = RussianDraughtsEngine(white, black, w_kings, b_kings, turn=turn)
        tokens = PDN_TOKEN_RE.findall(moves_text)
        _, main_line, branches, alt_wins, err = parse_variation_tree(tokens, 0, eng, 0)

        if err or len(main_line) < 2:
            failed += 1
            continue

        src_name = tags.get("Event") or default_source or path.stem
        author = tags.get("White") or tags.get("Author") or ""
        goal = "draw" if tags.get("Result") == "1/2-1/2" else "win"

        puzzle = {
            "id": f"{path.stem}-{idx}",
            "number": idx,
            "section": src_name,
            "sources": [f"{src_name} #{idx}"],
            "difficulty": 2 if len(main_line) >= 7 else 1,
            "author": author,
            "goal": goal,
            "white": white,
            "black": black,
            "whiteKings": w_kings,
            "blackKings": b_kings,
            "turn": turn,
            "mainLine": main_line,
            "branches": branches,
            "altWins": alt_wins,
        }
        p_tags, rating = compute_tags_and_rating(puzzle)
        puzzle["tags"] = p_tags
        puzzle["rating"] = rating
        puzzles.append(puzzle)

    return puzzles, failed


def load_json_file(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for p in data:
        if "sources" not in p:
            p["sources"] = [f"{p.get('section', path.stem)} №{p.get('number', '')}".strip()]
        p.setdefault("branches", [])
        p.setdefault("altWins", [])
        p_tags, rating = compute_tags_and_rating(p)
        p["tags"] = p_tags
        p["rating"] = rating
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Импорт PDN, объединение и дедупликация базы шашечных задач")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Собрать единую базу из .pdn и .json без дубликатов")
    p_build.add_argument("inputs", nargs="+", type=Path, help="Список .pdn и/или .json файлов")
    p_build.add_argument("--source", default="", help="Название книги/источника по умолчанию для PDN")
    p_build.add_argument("--out", type=Path, default=Path("master_puzzles.json"))

    args = parser.parse_args()
    by_hash: Dict[str, dict] = {}
    total_loaded = 0
    duplicates_merged = 0
    total_failed = 0

    for inp in args.inputs:
        if not inp.exists():
            print(f"[SKIP] Файл не найден: {inp}")
            continue
        if inp.suffix.lower() == ".pdn":
            items, failed = load_pdn_file(inp, args.source)
            total_failed += failed
            print(f"[PDN]  {inp.name}: валидно {len(items)}, пропущено {failed}")
        elif inp.suffix.lower() == ".json":
            items = load_json_file(inp)
            print(f"[JSON] {inp.name}: загружено {len(items)}")
        else:
            continue

        for p in items:
            total_loaded += 1
            h = position_hash(p)
            if h in by_hash:
                by_hash[h] = merge_puzzle(by_hash[h], p)
                duplicates_merged += 1
            else:
                by_hash[h] = p

    master = list(by_hash.values())
    args.out.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")

    with_branches = sum(1 for p in master if p.get("branches") or p.get("altWins"))
    print("\n" + "=" * 60)
    print(f"Всего загружено задач:      {total_loaded}")
    print(f"Объединено дубликатов:      {duplicates_merged}")
    print(f"Уникальных задач в базе:    {len(master)} -> {args.out}")
    print(f"С ветвлениями/вариантами:   {with_branches}")
    if total_failed:
        print(f"Отсеяно невалидных из PDN:  {total_failed}")


if __name__ == "__main__":
    main()