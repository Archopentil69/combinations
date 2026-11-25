#!/usr/bin/env python3
"""
Обогащение и расширение базы комбинаций Цукерника:
1. Восстановление названий 16 глав из chapters/ch_*.json.
2. Очистка и валидация дополнительных задач из reviews/*.json (включая рекурсивный
   lookahead при OCR-неоднозначностях и выделение встроенных вариантов в branches).
3. Подключение главы 16 (контрприёмы, tsukernik_ch16_black.json).
4. Объективная классификация по 4 уровням сложности (1..4) и рейтингу (800–2300).
5. Генерация содержательных тактических тегов и очистка OCR-мусора в поле author.
6. Сборка tsukernik_gold.json и puzzles_data.js.
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from audit_master import audit_puzzle, material_score
from draughts_solver import TacticalDraughtsSolver
from pdn_builder import position_hash
from tsukernik_validator import (
    FILES,
    Move,
    RussianDraughtsEngine,
    _parse_move_token,
    _char_cost,
    FILE_CONFUSIONS,
    RANK_CONFUSIONS,
    is_dark,
    match_move_with_repair,
    parse_solution_body,
    validate_puzzle_candidates,
)

ROOT = Path(__file__).resolve().parent
CHAPTERS_DIR = ROOT / "chapters"
REVIEWS_DIR = ROOT / "reviews"
BOOKS_DIR = ROOT / "books"
EXCLUDED_IDS = {"zlobinskiy-101", "zlobinskiy-187", "zlobinskiy-495", "zlobinskiy-1011", "sominskiy-224"}

CHAPTER_TITLES: Dict[str, str] = {
    "ch_01_a1_a7.json": "1. Финальный удар шашкой a1–a7",
    "ch_02_c1_c3.json": "2. Финальный удар шашкой c1–c3",
    "ch_03_e1_e5.json": "3. Финальный удар шашкой e1–e5",
    "ch_04_g1_g5.json": "4. Финальный удар шашкой g1–g5",
    "ch_05_b2.json": "5. Финальный удар шашкой b2",
    "ch_06_b4_b6.json": "6. Финальный удар шашкой b4–b6",
    "ch_07_d2.json": "7. Финальный удар шашкой d2",
    "ch_08_d4_d6.json": "8. Финальный удар шашкой d4–d6",
    "ch_09_f2.json": "9. Финальный удар шашкой f2",
    "ch_10_f4_f6.json": "10. Финальный удар шашкой f4–f6",
    "ch_11_h2.json": "11. Финальный удар шашкой h2",
    "ch_12_h4.json": "12. Финальный удар шашкой h4",
    "ch_13_h6.json": "13. Финальный удар шашкой h6",
    "ch_14_raznye_udary.json": "14. Разные удары",
    "ch_15_zapiranie_cugcvang_samoobl.json": "15. Запирание, цугцванг и самообложение",
    "ch_16_kontrpriemy.json": "16. Контрприёмы",
}

CHAPTER_RANGES: List[Tuple[int, int, str]] = [
    (1, 135, "1. Финальный удар шашкой a1–a7"),
    (136, 240, "2. Финальный удар шашкой c1–c3"),
    (241, 376, "3. Финальный удар шашкой e1–e5"),
    (377, 493, "4. Финальный удар шашкой g1–g5"),
    (494, 668, "5. Финальный удар шашкой b2"),
    (669, 821, "6. Финальный удар шашкой b4–b6"),
    (822, 972, "7. Финальный удар шашкой d2"),
    (973, 1065, "8. Финальный удар шашкой d4–d6"),
    (1066, 1290, "9. Финальный удар шашкой f2"),
    (1291, 1362, "10. Финальный удар шашкой f4–f6"),
    (1363, 1695, "11. Финальный удар шашкой h2"),
    (1696, 2141, "12. Финальный удар шашкой h4"),
    (2142, 2365, "13. Финальный удар шашкой h6"),
    (2366, 2569, "14. Разные удары"),
    (2570, 2638, "15. Запирание, цугцванг и самообложение"),
    (2639, 2900, "16. Контрприёмы"),
]


def infer_section(number: int, num_to_section: Dict[int, str]) -> str:
    if number in num_to_section:
        return num_to_section[number]
    for lo, hi, title in CHAPTER_RANGES:
        if lo <= number <= hi:
            return title
    return "14. Разные удары"


def clean_author_ocr(raw: str) -> str:
    """
    Отсеивает явный OCR-мусор в поле author (когда Tesseract читал кириллицу как латиницу),
    сохраняя читаемые фамилии, годы и ссылки.
    """
    s = str(raw or "").strip(" .,:;-—")
    if not s:
        return ""
    # Убираем обрывки примечаний и диаграммной обвязки
    if len(s) > 55:
        return ""
    # Если уже есть нормальная кириллица и почти нет латиницы — оставляем
    cyr = len(re.findall(r"[А-Яа-яЁё]", s))
    lat = len(re.findall(r"[A-Za-z]", s))
    if cyr >= 3 and lat <= 2:
        return s
    # Типичные мусорные строки (смесь цифр/спецсимволов/псевдо-латиницы)
    if re.search(r"[~|\\/{}<>]|(\b(?:fl|ll|JI|CT|KO|II|b\.|x\.|1\.)\b)", s):
        # Вытаскиваем хотя бы год, если он чистый, или возвращаем пустоту
        m_yr = re.search(r"\b(19[0-8]\d|189\d)\b", s)
        return f"Партия {m_yr.group(1)} г." if m_yr else ""
    if lat >= 3 and cyr == 0:
        # Иногда остаётся чисто латинское имя (например, зарубежный мастер) или мусор
        # Если слишком много заглавных букв вперемешку со строчными — это битый OCR кириллицы
        weird_caps = len(re.findall(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", s))
        if weird_caps >= 1 or any(ch in s for ch in ("'", '"', "0", "3", "4", "6")):
            m_yr = re.search(r"\b(19[0-8]\d|189\d)\b", s)
            return f"Партия {m_yr.group(1)} г." if m_yr else ""
    return s


def clean_raw_solution_text(rs: str) -> str:
    s = rs
    # Разделитель боя • или · -> :
    s = re.sub(r"([a-hA-H16])\s*[•·∙]\s*([a-hA-H16][1-8lISG])", r"\1:\2", s)
    s = re.sub(r"\bC\s*:\s*", "c:", s)
    s = re.sub(r"\b([a-h]):cl\b", r"\1:c1", s)
    s = re.sub(r"\bdG\b", "d6", s)
    s = re.sub(r"\b([a-h])G\b", r"\g<1>6", s)
    s = re.sub(r"\bf\s*:\s*df([1-8])", r"f:d\1", s)
    # Склеенный суффикс хода 'feSt' -> 'fe5!'
    s = re.sub(r"\b([a-h16]{2}[1-8S])t\b", r"\1!", s)
    # Склеенные номера ходов: 'b24.' -> 'b2 4.', 'g35.' -> 'g3 5.'
    s = re.sub(r"\b([a-h][1-8])([1-9]\.)\s*", r"\1 \2 ", s)
    return s


def candidate_move_matches(token: str, legal: List[Move], limit: int = 3) -> List[Move]:
    """Возвращает 1 точное совпадение или до `limit` лучших OCR-кандидатов."""
    mv, _ = match_move_with_repair(token, legal)
    if mv is not None:
        return [mv]

    parsed = _parse_move_token(token)
    if parsed is None or parsed["kind"] not in ("short", "start_sq_cap"):
        return []

    scored: List[Tuple[float, Move]] = []
    for m in legal:
        cost = (
            _char_cost(parsed["sf"], m.start_sq[0], FILE_CONFUSIONS)
            + _char_cost(parsed["ef"], m.end_sq[0], FILE_CONFUSIONS)
            + _char_cost(parsed["er"], m.end_sq[1], RANK_CONFUSIONS)
        )
        if parsed["kind"] == "start_sq_cap":
            cost += _char_cost(parsed["sr"], m.start_sq[1], RANK_CONFUSIONS)
        if m.is_capture != parsed["capture"]:
            cost += 0.15
        if cost <= 1.25:
            scored.append((cost, m))

    scored.sort(key=lambda x: x[0])
    return [m for _, m in scored[:limit]]


def solve_tokens_with_lookahead(
    eng: RussianDraughtsEngine,
    tokens: List[str],
    idx: int = 0,
    depth_budget: int = 24,
) -> Optional[List[str]]:
    """Рекурсивно сопоставляет цепочку OCR-токенов с легальными ходами при ничьей в стоимости OCR."""
    if idx >= len(tokens):
        return []
    if depth_budget <= 0:
        return None
    legal = eng.legal_moves()
    cands = candidate_move_matches(tokens[idx], legal, limit=3)
    for mv in cands:
        nxt = eng.clone()
        nxt.apply(mv)
        tail = solve_tokens_with_lookahead(nxt, tokens, idx + 1, depth_budget - 1)
        if tail is not None:
            return [mv.to_full_notation()] + tail
    return None


def _uses_king_privilege(line: List[str], king_sq: str, is_white: bool) -> bool:
    """Проверяет, что превращённая в дамку фигура реально ходит/бьёт по-дамочному в решении."""
    for idx, m_str in enumerate(line):
        if (idx % 2 == 0) != is_white:
            continue
        parts = m_str.replace("-", ":").split(":")
        if parts[0] == king_sq:
            # Дамка, если длинный ход (>2 клеток по диагонали) или взятие не со смежной клетки
            r0, c0 = int(parts[0][1]) - 1, "abcdefgh".index(parts[0][0])
            r1, c1 = int(parts[1][1]) - 1, "abcdefgh".index(parts[1][0])
            if "-" in m_str and abs(r1 - r0) > 1:
                return True
            if ":" in m_str and abs(r1 - r0) > 2:
                return True
            if "-" in m_str and ((is_white and r1 < r0) or (not is_white and r1 > r0)):
                return True
    return False


def try_recover_from_review(item: dict, section: str) -> Optional[dict]:
    rs = item.get("raw_solution", "")
    num = item.get("number")
    if not rs or not num:
        return None

    white = item.get("white", [])
    black = item.get("black", [])
    wk = item.get("whiteKings", [])
    bk = item.get("blackKings", [])
    if not white and not wk:
        return None
    if not black and not bk:
        return None

    cleaned = clean_raw_solution_text(rs)
    body = re.sub(r"^\s*\d{1,4}\s*[.,:]\s*", "", cleaned)

    variants = [body]
    # Вариант без встроенных под-вариантов вида '3... h:f2 ... 7. ab8x.'
    v1 = re.sub(r"\b\d+\s*\.\.\.[^.()]*?[xX×]\s*\.", " ", body)
    if v1 != body:
        variants.append(v1)
    v2 = re.sub(r"\b\d+\s*\.\.\.\s*.*?(?=\b\d+\.\s*[a-h16])", " ", body)
    if v2 != body and v2 != v1:
        variants.append(v2)

    for v in variants:
        parsed = parse_solution_body(num, v)
        if not parsed:
            continue
        # 1. Стандартная строгая проверка
        res = validate_puzzle_candidates(
            number=num,
            white=white,
            black=black,
            white_kings=wk,
            black_kings=bk,
            candidates=parsed.candidates,
            section=section,
            difficulty=1,
            author=parsed.author,
            goal="win",
        )
        if res.ok and res.puzzle:
            st, _ = audit_puzzle(res.puzzle)
            if st in ("gold", "ch16"):
                return res.puzzle

        # 2. Lookahead при равновероятных OCR-подменах (напр., be5! -> b4-c5 vs d4-e5)
        for cand in parsed.candidates:
            eng = RussianDraughtsEngine(white, black, wk, bk, turn="w")
            line = solve_tokens_with_lookahead(eng, cand.main_tokens)
            if line and len(line) >= 2:
                puzzle = {
                    "id": f"tsukernik-{num}",
                    "number": num,
                    "section": section,
                    "difficulty": 1,
                    "author": parsed.author,
                    "goal": "win",
                    "white": sorted(white),
                    "black": sorted(black),
                    "whiteKings": sorted(wk),
                    "blackKings": sorted(bk),
                    "turn": "w",
                    "mainLine": line,
                    "branches": [],
                }
                st, _ = audit_puzzle(puzzle)
                if st in ("gold", "ch16"):
                    return puzzle

        # 3. Если на диаграмме не была распознана дамка (templates=None), проверяем 1 дамку на существующем поле
        if not wk and not bk:
            for i, sq in enumerate(white):
                w2 = [s for k, s in enumerate(white) if k != i]
                for cand in parsed.candidates[:2]:
                    eng = RussianDraughtsEngine(w2, black, [sq], [], turn="w")
                    line = solve_tokens_with_lookahead(eng, cand.main_tokens)
                    if line and len(line) >= 3 and _uses_king_privilege(line, sq, True):
                        puzzle = {
                            "id": f"tsukernik-{num}",
                            "number": num,
                            "section": section,
                            "difficulty": 2,
                            "author": parsed.author,
                            "goal": "win",
                            "white": sorted(w2),
                            "black": sorted(black),
                            "whiteKings": [sq],
                            "blackKings": [],
                            "turn": "w",
                            "mainLine": line,
                            "branches": [],
                        }
                        st, _ = audit_puzzle(puzzle)
                        if st in ("gold", "ch16"):
                            return puzzle

            for i, sq in enumerate(black):
                b2 = [s for k, s in enumerate(black) if k != i]
                for cand in parsed.candidates[:2]:
                    eng = RussianDraughtsEngine(white, b2, [], [sq], turn="w")
                    line = solve_tokens_with_lookahead(eng, cand.main_tokens)
                    if line and len(line) >= 3 and _uses_king_privilege(line, sq, False):
                        puzzle = {
                            "id": f"tsukernik-{num}",
                            "number": num,
                            "section": section,
                            "difficulty": 2,
                            "author": parsed.author,
                            "goal": "win",
                            "white": sorted(white),
                            "black": sorted(b2),
                            "whiteKings": [],
                            "blackKings": [sq],
                            "turn": "w",
                            "mainLine": line,
                            "branches": [],
                        }
                        st, _ = audit_puzzle(puzzle)
                        if st in ("gold", "ch16"):
                            return puzzle
    return None


def enrich_puzzle_metadata(puzzle: dict, section: str, is_ch16: bool = False) -> dict:
    """
    Вычисляет объективный уровень сложности (1..4), рейтинг (800–2300),
    содержательные тактические теги и очищает поле author.
    """
    p = dict(puzzle)
    p["section"] = section
    p["author"] = clean_author_ocr(p.get("author", ""))
    p.setdefault("branches", [])
    p.setdefault("altWins", [])
    if not p.get("sources"):
        p["sources"] = [f"{section} №{p.get('number', '')}".strip()]

    eng = RussianDraughtsEngine(
        p.get("white", []),
        p.get("black", []),
        p.get("whiteKings", []),
        p.get("blackKings", []),
        turn=p.get("turn", "w"),
    )
    start_score, start_w, start_wk, start_b, start_bk = material_score(eng)

    main_line = p.get("mainLine", [])
    ply_count = len(main_line)
    max_single_cap = 0
    total_player_caps = 0
    quiet_late_moves = 0
    seen_player_capture = False
    has_mid_promo = False
    ends_with_promo_or_king = False
    min_player_material_drop = 0

    for idx, m_str in enumerate(main_line):
        legal = eng.legal_moves()
        mv, _ = match_move_with_repair(m_str, legal)
        if mv is None:
            break
        is_player = (eng.turn == p.get("turn", "w"))
        if is_player:
            if mv.is_capture:
                seen_player_capture = True
                max_single_cap = max(max_single_cap, len(mv.captured))
                total_player_caps += len(mv.captured)
            elif seen_player_capture and idx >= 4:
                # Тихий «роздых» уже после начала тактических взятий
                quiet_late_moves += 1
            if mv.promotes and mv.is_capture and len(mv.path) >= 3:
                has_mid_promo = True
            if idx >= len(main_line) - 2 and (mv.promotes or eng.board.get(mv.path[0], "").isupper()):
                ends_with_promo_or_king = True
        eng.apply(mv)
        sc, _, _, _, _ = material_score(eng)
        delta = (sc - start_score) if p.get("turn", "w") == "w" else (start_score - sc)
        min_player_material_drop = min(min_player_material_drop, delta)

    end_legal = eng.legal_moves()
    is_locked = len(end_legal) == 0

    # Теги (содержательные и селективные для фильтрации)
    tags = set()
    if p.get("goal") == "draw":
        tags.add("ничья")
    if is_ch16:
        tags.add("контрприёмы")
    if p.get("branches"):
        tags.add("варианты защиты")
    if p.get("altWins"):
        tags.add("альтернативный выигрыш")
    if has_mid_promo:
        tags.add("превращение в бою")
    if max_single_cap >= 4 or total_player_caps >= 5:
        tags.add("серия взятий (4+)")
    if min_player_material_drop <= -4:
        tags.add("жертва 4+ шашек")
    if quiet_late_moves >= 1:
        tags.add("роздых")
    if is_locked:
        tags.add("запирание")
    if start_wk > 0 or start_bk > 0 or (ends_with_promo_or_king and not has_mid_promo):
        tags.add("дамки")
    if ply_count <= 5:
        tags.add("быстрый удар")
    elif ply_count >= 11:
        tags.add("длинная комбинация")

    # Объективный скоринг сложности (idempotent, без опоры на предыдущий tier)
    sac_bonus = (
        4.5 if min_player_material_drop <= -5 else
        3.4 if min_player_material_drop <= -3 else
        2.2 if min_player_material_drop <= -2 else
        1.0 if min_player_material_drop <= -1 else 0.0
    )
    cap_bonus = min(3.5, max(0, max_single_cap - 1) * 0.9) + min(2.0, total_player_caps * 0.22)
    complexity = (
        ply_count * 0.95
        + quiet_late_moves * 2.8
        + len(p.get("branches", [])) * 1.8
        + sac_bonus
        + cap_bonus
        + (1.4 if has_mid_promo else 0.0)
        + (1.8 if p.get("goal") == "draw" else 0.0)
        + (1.6 if is_locked else 0.0)
        + (1.2 if (start_wk > 0 or start_bk > 0) else 0.0)
    )

    p["_complexity"] = complexity
    p["tags"] = sorted(tags)
    return p


def assign_calibrated_difficulty_tiers(puzzles: List[dict]) -> None:
    """
    Распределяет задачи по 4 уровням сложности и рейтингу (800–2300) по объективным квантилям.
    Для книги В. Ячейкина (1974), где нет объективно начальных/элементарных позиций,
    используются только уровни 2–4 (Средний / Сложный / Мастерский).
    """
    if not puzzles:
        return
    ordered = sorted(puzzles, key=lambda x: (x.get("_complexity", 0.0), len(x.get("mainLine", [])), x.get("number", 0)))
    n = len(ordered)

    yac_ordered = [p for p in ordered if str(p.get("id", "")).startswith("yacheykin-")]
    yac_pct_map: Dict[str, float] = {}
    yn = len(yac_ordered)
    for idx, p in enumerate(yac_ordered):
        yac_pct_map[p["id"]] = idx / max(1, yn - 1)

    for idx, p in enumerate(ordered):
        pct = idx / max(1, n - 1)
        comp = p.pop("_complexity", 10.0)
        if str(p.get("id", "")).startswith("yacheykin-"):
            ypct = yac_pct_map.get(p["id"], pct)
            if ypct < 0.30:
                tier = 2
            elif ypct < 0.65:
                tier = 3
            else:
                tier = 4
            # Рейтинг для Ячейкина от 1220 до 2300 (без начального уровня <1200)
            eff_pct = max(pct, 0.30 + ypct * 0.70)
            rating = int(round(820 + eff_pct * 1380 + min(80, comp * 1.8)))
        else:
            if pct < 0.25:
                tier = 1
            elif pct < 0.55:
                tier = 2
            elif pct < 0.82:
                tier = 3
            else:
                tier = 4
            rating = int(round(820 + pct * 1380 + min(80, comp * 1.8)))
        p["difficulty"] = tier
        p["rating"] = max(800, min(2300, rating))


def fix_rank8_or_rank1_kings(puzzle: dict) -> Optional[dict]:
    """
    В русских шашках простая белая шашка не может стоять на 8-й горизонтали,
    а простая чёрная — на 1-й. Если фигура на 8-й/1-й реально ходит как дамка
    в mainLine (например, №695), переводим её в дамки; иначе отбраковываем битую диаграмму.
    """
    p = dict(puzzle)
    w = list(p.get("white", []))
    b = list(p.get("black", []))
    wk = list(p.get("whiteKings", []))
    bk = list(p.get("blackKings", []))
    ml = p.get("mainLine", [])

    bad_w8 = [sq for sq in w if sq.endswith("8")]
    for sq in bad_w8:
        if _uses_king_privilege(ml, sq, True):
            w.remove(sq)
            if sq not in wk:
                wk.append(sq)
        else:
            return None

    bad_b1 = [sq for sq in b if sq.endswith("1")]
    for sq in bad_b1:
        if _uses_king_privilege(ml, sq, False):
            b.remove(sq)
            if sq not in bk:
                bk.append(sq)
        else:
            return None

    p["white"] = sorted(w)
    p["black"] = sorted(b)
    p["whiteKings"] = sorted(wk)
    p["blackKings"] = sorted(bk)
    return p


def is_complete_clean_combination(puzzle: dict, solver: TacticalDraughtsSolver) -> Tuple[bool, str]:
    """
    Проверяет, что комбинация:
      - не заканчивается ходом соперника (оборванный финал);
      - не обрывается на тихом ходе перед боем соперника или без результата;
      - в финале задачи на выигрыш позиция не является явно проигранной для белых.
    """
    ml = puzzle.get("mainLine", [])
    if len(ml) < 2:
        return False, "too_short"

    eng = RussianDraughtsEngine(
        puzzle.get("white", []),
        puzzle.get("black", []),
        puzzle.get("whiteKings", []),
        puzzle.get("blackKings", []),
        turn=puzzle.get("turn", "w"),
    )
    start_sc, _, _, _, _ = material_score(eng)
    last_mv: Optional[Move] = None
    for m_str in ml:
        mv, _ = match_move_with_repair(m_str, eng.legal_moves())
        if mv is None:
            return False, "illegal_move"
        eng.apply(mv)
        last_mv = mv

    # Оканчивается ходом соперника (очередь снова вернулась к игроку)
    if eng.turn == puzzle.get("turn", "w"):
        return False, "ends_on_opponent"

    end_sc, _, W, _, _ = material_score(eng)
    opp_legal = eng.legal_moves()

    if last_mv is not None and not last_mv.is_capture and len(opp_legal) > 0:
        if opp_legal[0].is_capture:
            return False, "quiet_before_opp_capture"
        if puzzle.get("goal") == "win" and end_sc <= start_sc and not last_mv.promotes:
            return False, "quiet_no_gain"

    # Проверка финальной позиции быстрым тактическим поиском (глубина 4)
    if puzzle.get("goal") == "win" and len(opp_legal) > 0:
        solver.clear()
        _, sc, _, _ = solver.search(eng, max_depth=4, time_limit_sec=0.04)
        white_sc = -sc if eng.turn == "b" else sc
        if white_sc <= -150:
            return False, "losing_final_position"

    return True, "ok"


def _recover_1cell_worker(item: dict) -> Optional[dict]:
    """Многопоточный ремонт 1 ошибочной/пропущенной шашки на диаграмме из reviews/ при наличии решения."""
    num = item.get("number")
    rs = item.get("raw_solution", "")
    if not isinstance(num, int) or num >= 2600 or not rs:
        return None
    w = list(item.get("white", []))
    b = list(item.get("black", []))
    if not w or not b:
        return None

    cleaned = clean_raw_solution_text(rs)
    body = re.sub(r"^\s*\d{1,4}\s*[.,:]\s*", "", cleaned)
    variants = [body]
    v1 = re.sub(r"\b\d+\s*\.\.\.[^.()]*?[xX×]\s*\.", " ", body)
    if v1 != body:
        variants.append(v1)

    parsed_cands = []
    author = ""
    for v in variants:
        p = parse_solution_body(num, v)
        if p:
            if not author:
                author = p.author
            parsed_cands.extend(p.candidates[:2])
    good_cands = [c for c in parsed_cands if len(c.main_tokens) >= 4]
    if not good_cands:
        return None

    tok_sqs = set()
    for c in good_cands[:2]:
        for t in c.main_tokens[:3]:
            for sq in re.findall(r"[a-h][1-8]", t.lower()):
                tok_sqs.add(sq)
    adj_sqs = set(tok_sqs)
    for sq in tok_sqs:
        r, col = int(sq[1]) - 1, FILES.index(sq[0])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, col + dc
                if 0 <= nr < 8 and 0 <= nc < 8 and is_dark(nr, nc):
                    adj_sqs.add(f"{FILES[nc]}{nr + 1}")

    occupied = set(w) | set(b)
    empty_near = [sq for sq in adj_sqs if sq not in occupied]

    hypotheses = []
    for i in range(len(w)):
        hypotheses.append(([x for k, x in enumerate(w) if k != i], b, [], []))
    for i in range(len(b)):
        hypotheses.append((w, [x for k, x in enumerate(b) if k != i], [], []))
    for i, sq in enumerate(w):
        if not sq.endswith("1"):
            hypotheses.append(([x for k, x in enumerate(w) if k != i], b + [sq], [], []))
    for i, sq in enumerate(b):
        if not sq.endswith("8"):
            hypotheses.append((w + [sq], [x for k, x in enumerate(b) if k != i], [], []))
    for sq in empty_near:
        if not sq.endswith("8"):
            hypotheses.append((w + [sq], b, [], []))
        if not sq.endswith("1"):
            hypotheses.append((w, b + [sq], [], []))

    solver = TacticalDraughtsSolver()
    for w2, b2, wk2, bk2 in hypotheses:
        for cand in good_cands[:2]:
            eng = RussianDraughtsEngine(w2, b2, wk2, bk2, turn="w")
            line = solve_tokens_with_lookahead(eng, cand.main_tokens)
            if line and len(line) >= 4:
                puz = {
                    "id": f"tsukernik-{num}",
                    "number": num,
                    "section": "Антология Цукерника",
                    "difficulty": 2,
                    "author": author,
                    "goal": "win",
                    "white": sorted(w2),
                    "black": sorted(b2),
                    "whiteKings": sorted(wk2),
                    "blackKings": sorted(bk2),
                    "turn": "w",
                    "mainLine": line,
                    "branches": [],
                }
                fixed = fix_rank8_or_rank1_kings(puz)
                if not fixed:
                    continue
                if audit_puzzle(fixed)[0] != "gold":
                    continue
                if is_complete_clean_combination(fixed, solver)[0]:
                    return fixed
    return None


def _solve_no_sol_worker(item: dict) -> Optional[dict]:
    """Автопоиск комбинации движком для диаграмм из reviews/ без текста решения."""
    num = item.get("number")
    if not isinstance(num, int) or num >= 2600:
        return None
    w = [sq for sq in item.get("white", []) if not sq.endswith("8")]
    b = [sq for sq in item.get("black", []) if not sq.endswith("1")]
    if len(w) < 2 or len(b) < 2:
        return None
    eng = RussianDraughtsEngine(w, b, [], [], turn="w")
    solver = TacticalDraughtsSolver()
    res = solver.find_combination(eng, max_depth=10, time_limit_sec=1.0, min_advantage=210, min_gap_over_second=110)
    if not res.get("found"):
        return None
    pv = res.get("pv", [])
    if len(pv) < 3:
        return None
    if len(pv) % 2 == 0:
        pv = pv[:-1]
    if len(pv) < 3:
        return None
    puz = {
        "id": f"tsukernik-{num}",
        "number": num,
        "section": "Антология Цукерника",
        "difficulty": 2,
        "author": "",
        "goal": "win",
        "white": sorted(w),
        "black": sorted(b),
        "whiteKings": [],
        "blackKings": [],
        "turn": "w",
        "mainLine": pv,
        "branches": [],
    }
    if audit_puzzle(puz)[0] != "gold":
        return None
    return puz if is_complete_clean_combination(puz, solver)[0] else None


def _auto_complete_truncated_worker(item: dict) -> Optional[dict]:
    """Дотягивает оборванные линии из tsukernik_suspect / full_master движком до чистого финала."""
    num = item.get("number")
    if not isinstance(num, int) or num >= 2600:
        return None
    eng = RussianDraughtsEngine(
        item.get("white", []),
        item.get("black", []),
        item.get("whiteKings", []),
        item.get("blackKings", []),
        turn=item.get("turn", "w"),
    )
    for m_str in item.get("mainLine", []):
        mv, _ = match_move_with_repair(m_str, eng.legal_moves())
        if mv is None:
            return None
        eng.apply(mv)

    solver = TacticalDraughtsSolver()
    extra: List[str] = []
    cur = eng.clone()
    for _ in range(6):
        legal = cur.legal_moves()
        if not legal:
            break
        if len(legal) == 1:
            mv = legal[0]
        else:
            mv, _, _, _ = solver.search(cur, max_depth=8, time_limit_sec=0.32)
        if not mv:
            break
        extra.append(mv.to_full_notation())
        cur.apply(mv)
        if cur.turn == "b":
            cand = dict(item)
            cand.pop("auditError", None)
            cand["mainLine"] = list(item.get("mainLine", [])) + extra
            fixed = fix_rank8_or_rank1_kings(cand)
            if fixed and audit_puzzle(fixed)[0] == "gold":
                if is_complete_clean_combination(fixed, solver)[0]:
                    _, end_sc, _, _ = solver.search(cur, max_depth=6, time_limit_sec=0.1)
                    if -end_sc >= 165 or not cur.legal_moves():
                        return fixed
    return None


def _precompute_alt_wins_worker(puzzle: dict) -> Tuple[str, List[dict]]:
    """
    Предвычисляет равноценные альтернативные первые ходы (afterPly=0),
    чтобы приложение засчитывало их мгновенно за 0 мс без обращения к Worker.
    """
    pid = puzzle["id"]
    existing = list(puzzle.get("altWins", []))
    if puzzle.get("goal", "win") != "win" or puzzle.get("turn", "w") != "w":
        return pid, existing
    ml = puzzle.get("mainLine", [])
    if not ml:
        return pid, existing

    eng0 = RussianDraughtsEngine(
        puzzle.get("white", []),
        puzzle.get("black", []),
        puzzle.get("whiteKings", []),
        puzzle.get("blackKings", []),
        turn="w",
    )
    legal0 = eng0.legal_moves()
    if len(legal0) <= 1:
        return pid, existing

    known_first = {ml[0]}
    for br in puzzle.get("branches", []):
        if br.get("afterPly") == 0 and br.get("moves"):
            known_first.add(br["moves"][0])
    for aw in existing:
        if aw.get("afterPly") == 0 and aw.get("moves"):
            known_first.add(aw["moves"][0])

    exp_mv = next((m for m in legal0 if m.to_full_notation() == ml[0]), None)
    if exp_mv is None:
        return pid, existing

    solver = TacticalDraughtsSolver()
    exp_eng = eng0.clone()
    exp_eng.apply(exp_mv)
    _, child_exp_sc, _, _ = solver.search(exp_eng, max_depth=8, time_limit_sec=0.06)
    exp_sc = -child_exp_sc
    if exp_sc < 180:
        return pid, existing

    found_alts = list(existing)
    for mv in legal0:
        if len(found_alts) >= 3:
            break
        nota = mv.to_full_notation()
        if nota in known_first:
            continue
        alt_eng = eng0.clone()
        alt_eng.apply(mv)
        _, child_sc, _, _ = solver.search(alt_eng, max_depth=8, time_limit_sec=0.06)
        sc = -child_sc
        both_mate = exp_sc >= 40_000 and sc >= 40_000
        close_win = sc >= 220 and sc >= exp_sc - 35
        if not (both_mate or close_win):
            continue

        # Углублённая проверка равноценного продолжения
        _, deep_sc, deep_pv, _ = solver.search(alt_eng, max_depth=10, time_limit_sec=0.14)
        sc2 = -deep_sc
        if not ((exp_sc >= 40_000 and sc2 >= 40_000) or (sc2 >= 220 and sc2 >= exp_sc - 35)):
            continue

        # Собираем чистую линию (нечётное число полуходов >= 3, финал без висящего взятия)
        probe = alt_eng.clone()
        line = [nota]
        for step_m in deep_pv[:10]:
            pmv = next((m for m in probe.legal_moves() if m.to_full_notation() == step_m), None)
            if pmv is None:
                break
            probe.apply(pmv)
            line.append(step_m)
            if probe.turn == "b" and len(line) >= 3 and not any(m.is_capture for m in probe.legal_moves()):
                break
        if len(line) < 3 or probe.turn != "b":
            continue
        if any(m.is_capture for m in probe.legal_moves()):
            continue

        cand = dict(puzzle)
        cand["mainLine"] = line
        cand["branches"] = []
        cand["altWins"] = []
        if audit_puzzle(cand)[0] != "gold":
            continue
        if not is_complete_clean_combination(cand, solver)[0]:
            continue
        _, end_sc_b, _, _ = solver.search(probe, max_depth=6, time_limit_sec=0.07)
        if -end_sc_b < 165 and probe.legal_moves():
            continue

        found_alts.append({"afterPly": 0, "moves": line})
        known_first.add(nota)

    return pid, found_alts


def main() -> None:
    # 1. Маппинг номеров задач -> реальные главы из chapters/ch_*.json (без 16-й главы контрприёмов)
    num_to_section: Dict[int, str] = {}
    for ch_file in sorted(CHAPTERS_DIR.glob("ch_*.json")):
        if "16_kontrpriemy" in ch_file.name:
            continue
        title = CHAPTER_TITLES.get(ch_file.name)
        if not title:
            continue
        arr = json.loads(ch_file.read_text(encoding="utf-8"))
        for x in arr:
            if isinstance(x.get("number"), int):
                num_to_section[x["number"]] = title

    by_hash: Dict[str, dict] = {}
    solver = TacticalDraughtsSolver()
    filtered_reasons: Counter[str] = Counter()

    def add_puzzle(raw_p: dict) -> bool:
        num = raw_p.get("number", 0)
        pid = str(raw_p.get("id", ""))
        if pid in EXCLUDED_IDS:
            filtered_reasons["excluded_id"] += 1
            return False
        is_tsuk = pid.startswith("tsukernik-") or not pid
        sec = infer_section(num, num_to_section) if is_tsuk else str(raw_p.get("section", "Доп. сборник"))

        if is_tsuk and ((isinstance(num, int) and num >= 2600) or sec.startswith("16.") or "контрприёмы" in raw_p.get("tags", [])):
            filtered_reasons["ch16_counter_tactics"] += 1
            return False

        fixed = fix_rank8_or_rank1_kings(raw_p)
        if fixed is None:
            filtered_reasons["illegal_rank8_or_rank1_piece"] += 1
            return False

        st, _ = audit_puzzle(fixed)
        if st != "gold":
            filtered_reasons["audit_failed"] += 1
            return False

        ok_fin, fin_reason = is_complete_clean_combination(fixed, solver)
        if not ok_fin:
            filtered_reasons[fin_reason] += 1
            return False

        h = position_hash(fixed)
        if h in by_hash:
            return False
        by_hash[h] = enrich_puzzle_metadata(fixed, sec, is_ch16=False)
        return True

    # 2. Базовые проверенные задачи Цукерника (без 16-й главы, битых диаграмм и старого Ячейкина)
    tsuk_gold_path = ROOT / "tsukernik_gold.json"
    if tsuk_gold_path.exists():
        for p in json.loads(tsuk_gold_path.read_text(encoding="utf-8")):
            if str(p.get("id", "")).startswith("yacheykin-"):
                continue
            add_puzzle(p)

    # 3. Чистые файлы chapters/ch_01..15
    for ch_file in sorted(CHAPTERS_DIR.glob("ch_*.json")):
        if "16_kontrpriemy" in ch_file.name:
            continue
        for p in json.loads(ch_file.read_text(encoding="utf-8")):
            add_puzzle(p)

    existing_tsuk_nums = {
        p["number"] for p in by_hash.values() if str(p.get("id", "")).startswith("tsukernik-")
    }

    # 4. Восстановление из reviews/ (главы 1–15): быстрый проход + 10-ядерный 1-cell repair & auto-solver
    recovered_count = 0
    review_with_sol: List[dict] = []
    review_no_sol: List[dict] = []
    seen_rev_keys = set()

    for rev_file in sorted(REVIEWS_DIR.glob("rev_*.json")):
        if "16_kontrpriemy" in rev_file.name:
            continue
        arr = json.loads(rev_file.read_text(encoding="utf-8"))
        for item in arr:
            num = item.get("number")
            if not isinstance(num, int) or num >= 2600 or num in existing_tsuk_nums:
                continue
            sec = infer_section(num, num_to_section)
            rec = try_recover_from_review(item, sec)
            if rec is not None and add_puzzle(rec):
                recovered_count += 1
                existing_tsuk_nums.add(num)
                continue
            rkey = (num, tuple(sorted(item.get("white", []))), tuple(sorted(item.get("black", []))))
            if rkey in seen_rev_keys:
                continue
            seen_rev_keys.add(rkey)
            if item.get("raw_solution"):
                review_with_sol.append(item)
            else:
                review_no_sol.append(item)

    workers = min(10, max(2, (os.cpu_count() or 4) - 2))
    print(f"Running {workers}-core recovery: {len(review_with_sol)} 1-cell repairs, {len(review_no_sol)} no-sol diagrams...")
    recovered_1cell = 0
    recovered_nosol = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_recover_1cell_worker, review_with_sol, chunksize=8):
            if res is not None and add_puzzle(res):
                recovered_1cell += 1
                existing_tsuk_nums.add(res["number"])
        for res in ex.map(_solve_no_sol_worker, review_no_sol, chunksize=4):
            if res is not None and add_puzzle(res):
                recovered_nosol += 1
                existing_tsuk_nums.add(res["number"])

    # 5. Многопоточная доигровка оборванных комбинаций из tsukernik_suspect.json и tsukernik_full_master.json
    trunc_candidates: List[dict] = []
    for fname in ("tsukernik_suspect.json", "tsukernik_full_master.json"):
        fpath = ROOT / fname
        if not fpath.exists():
            continue
        for item in json.loads(fpath.read_text(encoding="utf-8")):
            num = item.get("number")
            if isinstance(num, int) and num < 2600 and num not in existing_tsuk_nums:
                trunc_candidates.append(item)

    suspect_recovered = 0
    if trunc_candidates:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_auto_complete_truncated_worker, trunc_candidates, chunksize=6):
                if res is not None and add_puzzle(res):
                    suspect_recovered += 1
                    existing_tsuk_nums.add(res["number"])

    # 6. Подключение других книг (yacheykin_gold.json и books/*.json)
    extra_book_files: List[Path] = []
    yac_path = ROOT / "yacheykin_gold.json"
    if yac_path.exists():
        extra_book_files.append(yac_path)
    if BOOKS_DIR.exists():
        extra_book_files.extend(sorted(BOOKS_DIR.glob("*.json")))

    added_books: Counter[str] = Counter()
    for bf in extra_book_files:
        try:
            items = json.loads(bf.read_text(encoding="utf-8"))
            if isinstance(items, list):
                for p in items:
                    if add_puzzle(p):
                        added_books[bf.name] += 1
        except Exception as exc:
            print(f"Warning: failed loading {bf.name}: {exc}")

    # Сортировка: сначала Цукерник по номеру, затем Ячейкин и остальные книги
    def _sort_key(p: dict):
        pid = str(p.get("id", ""))
        group = 0 if pid.startswith("tsukernik-") else (1 if pid.startswith("yacheykin-") else 2)
        return (group, p.get("section", ""), p.get("number", 99999), pid)

    final_list = sorted(by_hash.values(), key=_sort_key)

    # 7. Предвычисление altWins (мгновенный приём равноценных первых ходов) на 10 ядрах
    print(f"Precomputing equivalent altWins across {len(final_list)} puzzles on {workers} cores...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        alt_map = dict(ex.map(_precompute_alt_wins_worker, final_list, chunksize=16))

    alt_puzzles_cnt = 0
    for p in final_list:
        alts = alt_map.get(p["id"], p.get("altWins", []))
        p["altWins"] = alts
        if alts:
            alt_puzzles_cnt += 1
            if "альтернативный выигрыш" not in p["tags"]:
                p["tags"] = sorted(set(p["tags"]) | {"альтернативный выигрыш"})

    assign_calibrated_difficulty_tiers(final_list)

    (ROOT / "tsukernik_gold.json").write_text(
        json.dumps(final_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "puzzles_data.js").write_text(
        "window.TSUKERNIK_PUZZLES = " + json.dumps(final_list, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

    diffs = Counter(p["difficulty"] for p in final_list)
    secs = Counter(p["section"] for p in final_list)
    tags = Counter(t for p in final_list for t in p["tags"])
    clean_authors = sum(1 for p in final_list if p.get("author"))

    print(
        f"Total unique clean puzzles: {len(final_list)} "
        f"(+reviews={recovered_count}, +1cell={recovered_1cell}, +nosol={recovered_nosol}, +suspect={suspect_recovered}, +books={dict(added_books)})"
    )
    print(f"Precomputed altWins on {alt_puzzles_cnt} puzzles")
    print("Filtered out:", dict(filtered_reasons))
    print(f"Difficulty tiers (1..4): {dict(sorted(diffs.items()))}")
    print(f"Clean authors: {clean_authors}")
    print("Sections:")
    for s, cnt in secs.most_common():
        print(f"  {s}: {cnt}")
    print("Tags:", dict(tags.most_common()))


if __name__ == "__main__":
    main()
