#!/usr/bin/env python3
"""
Пайплайн массовой оцифровки и строгой валидации книги
Э. Г. Цукерника «Антология шашечных комбинаций» (русские шашки 8x8).

Запуск:
  python tsukernik_validator.py pdf drafts_combinations.pdf --pages 12-29 --out ch_a1_a7.json
  python tsukernik_validator.py validate ch_a1_a7.json
"""

from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

FILES = "abcdefgh"
DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# ============================================================================
# 1. СТРОГИЙ ДВИЖОК РУССКИХ ШАШЕК 8x8
# ============================================================================

def sq_to_rc(sq: str) -> Tuple[int, int]:
    return int(sq[1]) - 1, FILES.index(sq[0])


def rc_to_sq(r: int, c: int) -> str:
    return f"{FILES[c]}{r + 1}"


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def is_dark(r: int, c: int) -> bool:
    return (r + c) % 2 == 0


@dataclass(frozen=True)
class Move:
    path: Tuple[Tuple[int, int], ...]
    captured: Tuple[Tuple[int, int], ...] = ()
    promotes: bool = False

    @property
    def is_capture(self) -> bool:
        return len(self.captured) > 0

    @property
    def start_sq(self) -> str:
        return rc_to_sq(*self.path[0])

    @property
    def end_sq(self) -> str:
        return rc_to_sq(*self.path[-1])

    def to_full_notation(self) -> str:
        if self.is_capture:
            return ":".join(rc_to_sq(r, c) for r, c in self.path)
        return f"{self.start_sq}-{self.end_sq}"


class RussianDraughtsEngine:
    def __init__(
        self,
        white: Sequence[str] = (),
        black: Sequence[str] = (),
        white_kings: Sequence[str] = (),
        black_kings: Sequence[str] = (),
        turn: str = "w",
    ):
        self.board: Dict[Tuple[int, int], str] = {}
        for sq in white:
            self.board[sq_to_rc(sq)] = "w"
        for sq in black:
            self.board[sq_to_rc(sq)] = "b"
        for sq in white_kings:
            self.board[sq_to_rc(sq)] = "W"
        for sq in black_kings:
            self.board[sq_to_rc(sq)] = "B"
        self.turn = turn

    def clone(self) -> "RussianDraughtsEngine":
        eng = RussianDraughtsEngine(turn=self.turn)
        eng.board = dict(self.board)
        return eng

    @staticmethod
    def _is_own(piece: str, color: str) -> bool:
        return piece.lower() == color

    @staticmethod
    def _is_enemy(piece: str, color: str) -> bool:
        return piece.lower() != color

    @staticmethod
    def _promo_rank(color: str) -> int:
        return 7 if color == "w" else 0

    def legal_moves(self, color: Optional[str] = None) -> List[Move]:
        color = color or self.turn
        captures: List[Move] = []

        for (r, c), piece in list(self.board.items()):
            if not self._is_own(piece, color):
                continue
            is_king = piece.isupper()
            working_board = dict(self.board)
            del working_board[(r, c)]

            raw_chains = self._find_captures(working_board, (r, c), color, is_king, set())
            for tail_path, cap_list, final_king in raw_chains:
                captures.append(
                    Move(
                        path=((r, c),) + tail_path,
                        captured=cap_list,
                        promotes=(not is_king and final_king),
                    )
                )

        if captures:
            return list(dict.fromkeys(captures))

        quiet: List[Move] = []
        for (r, c), piece in self.board.items():
            if not self._is_own(piece, color):
                continue
            if piece.islower():
                dr = 1 if color == "w" else -1
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc) and (nr, nc) not in self.board:
                        quiet.append(
                            Move(
                                path=((r, c), (nr, nc)),
                                promotes=(nr == self._promo_rank(color)),
                            )
                        )
            else:
                for dr, dc in DIRS:
                    nr, nc = r + dr, c + dc
                    while in_bounds(nr, nc) and (nr, nc) not in self.board:
                        quiet.append(Move(path=((r, c), (nr, nc))))
                        nr += dr
                        nc += dc
        return quiet

    def _find_captures(
        self,
        board: Dict[Tuple[int, int], str],
        pos: Tuple[int, int],
        color: str,
        is_king: bool,
        captured: Set[Tuple[int, int]],
    ) -> List[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], bool]]:
        r, c = pos
        results: List[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], bool]] = []

        if not is_king:
            for dr, dc in DIRS:
                mr, mc = r + dr, c + dc
                lr, lc = r + 2 * dr, c + 2 * dc
                if not in_bounds(lr, lc):
                    continue
                target = board.get((mr, mc))
                if target is None or not self._is_enemy(target, color):
                    continue
                if (mr, mc) in captured or (lr, lc) in board:
                    continue

                next_captured = captured | {(mr, mc)}
                next_king = lr == self._promo_rank(color)
                sub = self._find_captures(board, (lr, lc), color, next_king, next_captured)
                if sub:
                    for tail_path, cap_list, final_king in sub:
                        results.append(
                            (((lr, lc),) + tail_path, ((mr, mc),) + cap_list, final_king)
                        )
                else:
                    results.append((((lr, lc),), ((mr, mc),), next_king))
        else:
            for dr, dc in DIRS:
                sr, sc = r + dr, c + dc
                while in_bounds(sr, sc) and (sr, sc) not in board:
                    sr += dr
                    sc += dc
                if not in_bounds(sr, sc):
                    continue
                target = board.get((sr, sc))
                if target is None or not self._is_enemy(target, color) or (sr, sc) in captured:
                    continue

                landings: List[Tuple[int, int]] = []
                lr, lc = sr + dr, sc + dc
                while in_bounds(lr, lc) and (lr, lc) not in board:
                    landings.append((lr, lc))
                    lr += dr
                    lc += dc

                if not landings:
                    continue

                next_captured = captured | {(sr, sc)}
                continuing: List[
                    Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], bool]
                ] = []
                for land in landings:
                    sub = self._find_captures(board, land, color, True, next_captured)
                    for tail_path, cap_list, final_king in sub:
                        continuing.append(
                            ((land,) + tail_path, ((sr, sc),) + cap_list, final_king)
                        )

                if continuing:
                    results.extend(continuing)
                else:
                    for land in landings:
                        results.append(((land,), ((sr, sc),), True))

        return results

    def apply(self, move: Move) -> None:
        piece = self.board.pop(move.path[0])
        for cap in move.captured:
            self.board.pop(cap, None)
        if move.promotes:
            piece = piece.upper()
        self.board[move.path[-1]] = piece
        self.turn = "b" if self.turn == "w" else "w"


# ============================================================================
# 2. УМНЫЙ ПАРСЕР НОТАЦИИ И АВТОРЕМОНТ OCR-ОПЕЧАТОК
# ============================================================================

FILE_CONFUSIONS: Dict[str, str] = {
    "a": "d4o",
    "b": "6dh8",
    "c": "eo",
    "d": "ab6",
    "e": "co",
    "f": "1lti",
    "g": "9q",
    "h": "bn1",
}
RANK_CONFUSIONS: Dict[str, str] = {
    "1": "liI7f",
    "2": "z",
    "3": "8з",
    "4": "1a",
    "5": "6sSб",
    "6": "58bб",
    "7": "1",
    "8": "36bS",
}


def normalize_moves_text(text: str) -> str:
    """Чистит опечатки OCR внутри строки с ходами."""
    cyr_lower = str.maketrans({
        "а": "a", "с": "c", "е": "e", "х": "x", "×": "x", "—": "-", "–": "-", ";": ":",
    })
    s = text.translate(cyr_lower)
    s = re.sub(r"\s+", " ", s)
    # Пробелы вокруг двоеточия: 'h : f2' -> 'h:f2'
    s = re.sub(r"\s*:\s*", ":", s)

    # Расщеплённые буквы 'd' -> 'cI'/'cl', 'h' -> '11', 'cb4'/'ab4' -> 'eM'/'aM'
    s = re.sub(r"\bc[Il]:", "d:", s)
    s = re.sub(r"\b([a-h])c[Il]([1-8S])\b", r"\1d\2", s)
    s = re.sub(r"\bc[Il]([a-h][1-8S])\b", r"d\1", s)
    s = re.sub(r"\b11:", "h:", s)
    s = re.sub(r"\b([a-h]):11([2468])\b", r"\1:h\2", s)
    s = re.sub(r"\beM\b", "cb4", s)
    s = re.sub(r"\baM\b", "ab4", s)

    # Пробел внутри взятия после буквы: 'c:e 1' -> 'c:e1'
    s = re.sub(r"\b([a-h]:[a-h])\s+([1-8lIS])\b", r"\1\2", s)

    # Потерянное двоеточие с пробелом: 'c f6' -> 'c:f6', 'e el' -> 'e:el', 'h d4' -> 'h:d4'
    s = re.sub(
        r"(?<![A-Za-zА-Яа-я0-9:])([a-h])\s+([a-h][1-8lIS])(?![A-Za-zА-Яа-я0-9])",
        r"\1:\2",
        s,
    )

    # Слипшиеся номера ходов: 'b2S.' -> 'b2 5.', 'b24.a:c7' -> 'b2 4. a:c7'
    s = re.sub(r"([a-h16][1-8lIS])S\.", r"\1 5. ", s)
    s = re.sub(r"([a-h16][1-8lIS])(\d{1,2}\.)", r"\1 \2 ", s)
    s = re.sub(r"(\d{1,2}\.)(?=[a-h16])", r"\1 ", s)

    # Слипшиеся пары ходов: 'eglcg5' -> 'egl cg5', 'bc3d:b2' -> 'bc3 d:b2'
    glue_pat = re.compile(
        r"(?<!:)\b([a-h16]:?[a-h16][1-8lIS])([a-h16]:?[a-h16][1-8lIS])\b(?!:)"
    )
    for _ in range(3):
        s = glue_pat.sub(r"\1 \2", s)

    # Частые OCR-подмены в клетках
    s = re.sub(r"\b([a-h]):1([2468])\b", r"\1:f\2", s)
    s = re.sub(r"\b1:([a-h][1-8S])\b", r"f:\1", s)
    s = re.sub(r"\b([a-h])6([2468])\b", r"\1b\2", s)
    s = re.sub(r"\b([a-h]):6([2468])\b", r"\1:b\2", s)
    s = re.sub(r"\b([a-h])([a-h])[lI]\b", r"\1\g<2>1", s)
    s = re.sub(r"\b([a-h]):([a-h])[lI]\b", r"\1:\g<2>1", s)
    return s.strip()


def _char_cost(got: str, expected: str, table: Dict[str, str]) -> float:
    if got == expected:
        return 0.0
    if got in table.get(expected, "") or expected in table.get(got, ""):
        return 0.4
    return 1.25


def _parse_move_token(raw: str) -> Optional[dict]:
    tok = raw.strip(".,!?;xX= ")
    if not tok:
        return None

    if tok.count(":") >= 2:
        parts = tok.split(":")
        if all(re.fullmatch(r"[a-h][1-8]", p) for p in parts):
            return {"kind": "chain", "squares": parts, "capture": True, "raw": raw}

    m = re.fullmatch(r"([a-h][1-8])-([a-h][1-8])", tok)
    if m:
        return {"kind": "full_quiet", "start": m.group(1), "end": m.group(2), "capture": False, "raw": raw}

    m = re.fullmatch(r"([a-h16])([1-8lIS]):([a-h16])([1-8lIS])", tok)
    if m:
        return {
            "kind": "start_sq_cap",
            "sf": m.group(1), "sr": m.group(2),
            "ef": m.group(3), "er": m.group(4),
            "capture": True, "raw": raw,
        }

    m = re.fullmatch(r"([a-h16]):([a-h16])([1-8lIS])", tok)
    if m:
        return {
            "kind": "short",
            "sf": m.group(1), "ef": m.group(2), "er": m.group(3),
            "capture": True, "raw": raw,
        }

    m = re.fullmatch(r"([a-h16])([a-h16])([1-8lIS])", tok)
    if m:
        return {
            "kind": "short",
            "sf": m.group(1), "ef": m.group(2), "er": m.group(3),
            "capture": False, "raw": raw,
        }
    return None


def match_move_with_repair(
    token: str, legal: List[Move]
) -> Tuple[Optional[Move], Optional[str]]:
    if not legal:
        return None, "Нет легальных ходов в позиции"

    parsed = _parse_move_token(token)
    if parsed is None:
        return None, f"Нераспознанный формат токена '{token}'"

    exact: List[Move] = []
    for mv in legal:
        if parsed["kind"] == "chain":
            if mv.is_capture and [rc_to_sq(r, c) for r, c in mv.path] == parsed["squares"]:
                exact.append(mv)
        elif parsed["kind"] == "full_quiet":
            if not mv.is_capture and mv.start_sq == parsed["start"] and mv.end_sq == parsed["end"]:
                exact.append(mv)
        elif parsed["kind"] == "start_sq_cap":
            if (
                mv.is_capture
                and mv.start_sq == parsed["sf"] + parsed["sr"]
                and mv.end_sq == parsed["ef"] + parsed["er"]
            ):
                exact.append(mv)
        elif parsed["kind"] == "short":
            if (
                mv.start_sq[0] == parsed["sf"]
                and mv.end_sq == parsed["ef"] + parsed["er"]
                and (mv.is_capture == parsed["capture"] or legal[0].is_capture)
            ):
                exact.append(mv)

    if len(exact) == 1:
        note = None
        if parsed["kind"] == "short" and legal[0].is_capture and not parsed["capture"]:
            note = f"Восстановлено двоеточие: '{token}' -> '{exact[0].to_full_notation()}'"
        return exact[0], note

    if len(exact) > 1:
        return exact[0], f"Неоднозначность для '{token}', выбран '{exact[0].to_full_notation()}'"

    if parsed["kind"] in ("short", "start_sq_cap"):
        scored: List[Tuple[float, Move]] = []
        for mv in legal:
            cost = (
                _char_cost(parsed["sf"], mv.start_sq[0], FILE_CONFUSIONS)
                + _char_cost(parsed["ef"], mv.end_sq[0], FILE_CONFUSIONS)
                + _char_cost(parsed["er"], mv.end_sq[1], RANK_CONFUSIONS)
            )
            if parsed["kind"] == "start_sq_cap":
                cost += _char_cost(parsed["sr"], mv.start_sq[1], RANK_CONFUSIONS)
            if mv.is_capture != parsed["capture"]:
                cost += 0.15
            scored.append((cost, mv))

        scored.sort(key=lambda x: x[0])
        best_cost, best_mv = scored[0]
        second_cost = scored[1][0] if len(scored) > 1 else 99.0

        if best_cost <= 1.25 and (second_cost - best_cost) >= 0.35:
            return best_mv, f"OCR-ремонт: '{token}' -> '{best_mv.to_full_notation()}'"

        if len(legal) == 1 and legal[0].is_capture and best_cost <= 2.4:
            return best_mv, f"Вынужденный бой (OCR '{token}' -> '{best_mv.to_full_notation()}')"

    return None, (
        f"Ход '{token}' не найден среди легальных: "
        + ", ".join(m.to_full_notation() for m in legal[:8])
    )


# ============================================================================
# 3. ПОСТРОЧНЫЙ МОНОТОННЫЙ ПАРСЕР ОТВЕТОВ
# ============================================================================

@dataclass
class SolutionCandidate:
    main_tokens: List[str]
    branches: List[Tuple[int, List[str]]]


@dataclass
class ParsedSolution:
    number: int
    author: str
    candidates: List[SolutionCandidate]
    raw_text: str


MOVE_TOKEN_RE = re.compile(
    r"(?<![A-Za-zА-Яа-я0-9])"
    r"(?:[a-h][1-8](?::[a-h][1-8]){2,}|[a-h][1-8]-[a-h][1-8]|"
    r"[a-h16][1-8lIS]?:[a-h16][1-8lIS]|[a-h16]{2}[1-8lIS])[!?\u00d7xX=]*"
    r"(?![A-Za-zА-Яа-я0-9])"
)

COMMENT_CUT_RE = re.compile(
    r"\b(?:Плохо|Нельзя|Белые\s+также|Возможно\s+также|Неудачн|Ошибк|Если\s+вместо|"
    r"B03MO|OlllH6K|HeY\)1\.a)\b",
    re.IGNORECASE,
)

TACTICAL_LINE_RE = re.compile(
    r"(?:Тактические\s+элементы|ФИНАЛЬНЫЙ\s+УДАР|Перевод\s+(?:черной|белой)|Удаление\s+черной|"
    r"nepeBo|Y,?[lI]\.?aJIeHHe|TaKTH4eCK)",
    re.IGNORECASE,
)


def _decode_ocr_int(tok: str) -> Optional[int]:
    trans = str.maketrans({
        "O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "b": "6",
    })
    d = tok.translate(trans)
    return int(d) if d.isdigit() else None


def _parse_single_candidate(moves_part: str) -> Optional[SolutionCandidate]:
    moves_part = re.sub(r"\(см\.\s*также[^)]*\)", "", moves_part, flags=re.IGNORECASE)
    moves_part = re.sub(r"\([eс]M\.\s*TaK[^)]*\)", "", moves_part, flags=re.IGNORECASE)

    branches: List[Tuple[int, List[str]]] = []
    main_only_parts: List[str] = []
    idx = 0
    ply_counter = 0

    for m_par in re.finditer(r"\(([^()]+)\)", moves_part):
        before = moves_part[idx:m_par.start()]
        main_only_parts.append(before)
        ply_counter += len(MOVE_TOKEN_RE.findall(before))

        inside = m_par.group(1).strip()
        m_br = re.match(r"^(\d+)\s*\.\.\.\s*(.*)$", inside)
        if m_br:
            move_no = int(m_br.group(1))
            br_tokens = MOVE_TOKEN_RE.findall(m_br.group(2))
            if br_tokens:
                branches.append((2 * move_no - 1, br_tokens))
        else:
            alt_toks = MOVE_TOKEN_RE.findall(inside)
            if len(alt_toks) == 1 and ":" in alt_toks[0] and ply_counter > 0:
                branches.append((ply_counter - 1, alt_toks))
        idx = m_par.end()

    main_only_parts.append(moves_part[idx:])
    main_str = " ".join(main_only_parts)

    m_cut = COMMENT_CUT_RE.search(main_str)
    if m_cut:
        main_str = main_str[: m_cut.start()]

    main_tokens: List[str] = []
    prose_streak = 0
    for word in main_str.split():
        if re.fullmatch(r"\d{1,2}\.", word):
            prose_streak = 0
            continue
        m_tok = MOVE_TOKEN_RE.fullmatch(word.rstrip(".,"))
        if m_tok:
            tok = m_tok.group(0)
            main_tokens.append(tok)
            prose_streak = 0
            if tok.endswith(("x", "X", "=")):
                break
        else:
            prose_streak += 1
            if prose_streak >= 4 and len(main_tokens) >= 3:
                break

    if len(main_tokens) < 2:
        return None
    return SolutionCandidate(main_tokens=main_tokens, branches=branches)


def parse_solution_body(num: int, raw_body: str) -> Optional[ParsedSolution]:
    norm = normalize_moves_text(raw_body)
    starts = list(re.finditer(r"(?:^|[\s(])(?:1|[IlJ])[.,']{1,2}\s*(?=[a-h16])", norm))
    if not starts:
        return None

    author = norm[: starts[0].start()].strip(" .—-:,")
    scored_cands: List[Tuple[int, SolutionCandidate]] = []

    for i, m in enumerate(starts):
        frag = norm[m.start():]
        cand = _parse_single_candidate(frag)
        if not cand:
            continue
        prio = 0
        first_tok = cand.main_tokens[0]
        if "?" in first_tok:
            prio -= 10
        if "!" in first_tok:
            prio += 5
        prefix = norm[max(0, m.start() - 25): m.start()].lower()
        if any(w in prefix for w in ("после", "плохо", "нельзя", "если", "ошибк")):
            prio -= 8
        prio += i
        scored_cands.append((prio, cand))

    if not scored_cands:
        return None
    scored_cands.sort(key=lambda x: x[0], reverse=True)
    return ParsedSolution(
        number=num,
        author=author,
        candidates=[c for _, c in scored_cands],
        raw_text=f"{num}. {raw_body}".strip(),
    )


def extract_solutions_from_lines(
    lines: List[str], target_numbers: Set[int]
) -> Dict[int, ParsedSolution]:
    if not target_numbers:
        return {}
    min_target, max_target = min(target_numbers), max(target_numbers)

    raw_chunks: Dict[int, List[str]] = {}
    current_num: Optional[int] = None
    last_num = min_target - 1
    in_tactical_header = False

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue

        if TACTICAL_LINE_RE.search(s):
            in_tactical_header = True
            current_num = None
            continue

        m_start = re.match(r"^([0-9OIlSBb]{1,4})\s*[.,:]\s*(.*)$", s)
        if m_start:
            cand_n = _decode_ocr_int(m_start.group(1))
            rest = m_start.group(2).strip()

            if in_tactical_header and cand_n is not None and cand_n <= 15:
                if not re.search(r"(?:1|[IlJ])\.\s*[a-h16]", rest):
                    continue

            if cand_n is not None and min_target <= cand_n <= max_target:
                is_monotonic = (
                    (current_num is None and cand_n >= last_num + 1 and cand_n <= last_num + 12)
                    or (current_num is not None and current_num < cand_n <= current_num + 8)
                )
                if is_monotonic:
                    looks_like_continuation_move = (
                        current_num is not None
                        and cand_n <= 15
                        and bool(re.match(r"^[a-h16]:?[a-h16][1-8lIS]", rest))
                        and not re.search(r"\b(?:1|[IlJ])\.\s*[a-h16]", rest)
                        and not (
                            raw_chunks.get(current_num)
                            and re.search(r"[xX×=]\.?$", raw_chunks[current_num][-1].strip())
                        )
                    )
                    if not looks_like_continuation_move:
                        in_tactical_header = False
                        current_num = cand_n
                        last_num = cand_n
                        raw_chunks.setdefault(cand_n, [])
                        if rest:
                            raw_chunks[cand_n].append(rest)
                        continue

        if current_num is not None:
            raw_chunks[current_num].append(s)

    solutions: Dict[int, ParsedSolution] = {}
    for num, parts in raw_chunks.items():
        joined = ""
        for p in parts:
            if joined.endswith("-") and not joined.endswith(" -"):
                joined = joined[:-1] + p
            else:
                joined = (joined + " " + p).strip()
        parsed = parse_solution_body(num, joined)
        if parsed is not None:
            solutions[num] = parsed
    return solutions


# ============================================================================
# 4. РАСПОЗНАВАНИЕ ДИАГРАММ + КЛАСТЕРНАЯ СОРТИРОВКА РЯДОВ
# ============================================================================

@dataclass
class DetectedBoard:
    page: int
    bbox: Tuple[int, int, int, int]
    raw_number: Optional[int]
    raw_confident: bool
    number: Optional[int]
    difficulty: int
    goal: str
    has_quiet_tag: bool
    white: List[str]
    black: List[str]
    white_kings: List[str] = field(default_factory=list)
    black_kings: List[str] = field(default_factory=list)
    crop_img: Optional[np.ndarray] = None


def load_templates(template_dir: Optional[Path]) -> Dict[str, np.ndarray]:
    templates: Dict[str, np.ndarray] = {}
    if not template_dir or not template_dir.exists():
        return templates
    for key in ("w", "b", "W", "B", "empty"):
        p = template_dir / f"{key}.png"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[key] = img
    return templates


def trim_outer_border(board_gray: np.ndarray) -> np.ndarray:
    h, w = board_gray.shape
    max_tx, max_ty = max(2, int(w * 0.065)), max(2, int(h * 0.065))

    top = 0
    while top < max_ty and float(np.mean(board_gray[top, :])) < 165:
        top += 1
    bottom = h - 1
    while (h - 1 - bottom) < max_ty and float(np.mean(board_gray[bottom, :])) < 165:
        bottom -= 1
    left = 0
    while left < max_tx and float(np.mean(board_gray[:, left])) < 165:
        left += 1
    right = w - 1
    while (w - 1 - right) < max_tx and float(np.mean(board_gray[:, right])) < 165:
        right -= 1

    if top == 0:
        top = int(h * 0.022)
    if bottom == h - 1:
        bottom = h - 1 - int(h * 0.022)
    if left == 0:
        left = int(w * 0.022)
    if right == w - 1:
        right = w - 1 - int(w * 0.022)

    return board_gray[top:bottom + 1, left:right + 1]


def cell_metrics(cell_gray: np.ndarray, otsu_t: float) -> Tuple[float, float, float, float]:
    h, w = cell_gray.shape
    patch = cell_gray[int(h * 0.12):int(h * 0.88), int(w * 0.12):int(w * 0.88)]
    ph, pw = patch.shape
    if ph < 8 or pw < 8:
        return 0.0, 0.0, 0.0, 0.0

    bright = (patch > otsu_t).astype(np.uint8)
    dark = (patch <= otsu_t).astype(np.uint8)

    dist_b = cv2.distanceTransform(bright, cv2.DIST_L2, 5)
    dist_d = cv2.distanceTransform(dark, cv2.DIST_L2, 5)

    y1, y2 = int(ph * 0.12), int(ph * 0.88)
    x1, x2 = int(pw * 0.12), int(pw * 0.88)

    dark_ratio = float(np.mean(dark[y1:y2, x1:x2]))
    r_dark = float(np.max(dist_d[y1:y2, x1:x2])) / pw
    r_bright = float(np.max(dist_b[y1:y2, x1:x2])) / pw
    open_white = float(np.mean(dist_b[y1:y2, x1:x2] >= pw * 0.085))
    return dark_ratio, r_dark, r_bright, open_white


def classify_cell(
    cell_gray: np.ndarray, otsu_t: float, templates: Dict[str, np.ndarray]
) -> str:
    if templates:
        best_label, best_score = ".", -1.0
        for label, tmpl in templates.items():
            resized = cv2.resize(cell_gray, (tmpl.shape[1], tmpl.shape[0]))
            res = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score, best_label = score, label
        return "." if best_label == "empty" else best_label

    dark_ratio, r_dark, r_bright, open_white = cell_metrics(cell_gray, otsu_t)

    if dark_ratio >= 0.34 and r_dark >= 0.125:
        return "b"
    if r_bright >= 0.095 and open_white >= 0.025:
        return "w"
    return "."


def sort_boxes_by_rows(raw_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Сортирует доски строго по рядам сверху вниз, а внутри ряда слева направо.
    Устойчива к наклону скана (без артефактов целочисленного деления).
    """
    if not raw_boxes:
        return []
    med_h = float(np.median([b[3] for b in raw_boxes]))
    by_y = sorted(raw_boxes, key=lambda b: b[1])
    rows: List[List[Tuple[int, int, int, int]]] = []
    for box in by_y:
        if not rows:
            rows.append([box])
        else:
            row_y = float(np.mean([b[1] for b in rows[-1]]))
            if abs(box[1] - row_y) <= med_h * 0.45:
                rows[-1].append(box)
            else:
                rows.append([box])

    ordered: List[Tuple[int, int, int, int]] = []
    for row in rows:
        row.sort(key=lambda b: b[0])
        ordered.extend(row)
    return ordered


DIAG_CODE_WORD_RE = re.compile(r"^[\d.*=()'\-\"`$TXТХIx]+$", re.IGNORECASE)


def detect_boards_and_text_lines(
    page: fitz.Page, page_no: int, templates: Dict[str, np.ndarray], dpi: int = 250
) -> Tuple[List[DetectedBoard], List[str]]:
    scale = dpi / 72.0
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_page, w_page = gray.shape
    raw_boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h else 0
        if 0.88 <= aspect <= 1.12 and (w_page * 0.18) <= w <= (w_page * 0.36):
            raw_boxes.append((x, y, w, h))

    raw_boxes = sort_boxes_by_rows(raw_boxes)
    words = page.get_text("words", sort=True)
    detected: List[DetectedBoard] = []

    for x, y, w, h in raw_boxes:
        board_gray = gray[y:y + h, x:x + w]
        inner = trim_outer_border(board_gray)
        otsu_raw, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_t = float(np.clip(otsu_raw, 130, 210))

        ih, iw = inner.shape
        cw, ch = iw / 8.0, ih / 8.0

        white, black, w_kings, b_kings = [], [], [], []
        for row_img in range(8):
            r = 7 - row_img
            for c in range(8):
                if not is_dark(r, c):
                    continue
                y1, y2 = int(row_img * ch), int((row_img + 1) * ch)
                x1, x2 = int(c * cw), int((c + 1) * cw)
                label = classify_cell(inner[y1:y2, x1:x2], otsu_t, templates)
                sq = rc_to_sq(r, c)
                if label == "w":
                    white.append(sq)
                elif label == "b":
                    black.append(sq)
                elif label == "W":
                    w_kings.append(sq)
                elif label == "B":
                    b_kings.append(sq)

        px0, px1 = (x - w * 0.05) / scale, (x + w * 1.05) / scale
        py0, py1 = (y - h * 0.18) / scale, (y + h * 0.02) / scale
        hdr_words = [
            wt[4] for wt in words
            if px0 <= (wt[0] + wt[2]) / 2 <= px1 and py0 <= (wt[1] + wt[3]) / 2 <= py1
        ]
        hdr = " ".join(hdr_words)

        raw_num: Optional[int] = None
        confident = False
        for wt in hdr_words:
            if "." in wt or "(" in wt or ")" in wt:
                continue
            m_star = re.match(r"^(\d{1,4})([=*\-\'\"`]+)$", wt)
            if m_star:
                raw_num = int(m_star.group(1))
                confident = True
                break
        if raw_num is None:
            for wt in hdr_words:
                if "." in wt or "(" in wt or ")" in wt:
                    continue
                m_plain = re.fullmatch(r"(\d{1,4})", wt)
                if m_plain:
                    raw_num = int(m_plain.group(1))
                    break

        stars = min(3, max(1, hdr.count("*")))
        goal = "draw" if "=" in hdr else "win"
        has_tx = bool(re.search(r"\b(?:TX|ТХ)\b", hdr, flags=re.IGNORECASE))

        detected.append(
            DetectedBoard(
                page=page_no,
                bbox=(x, y, w, h),
                raw_number=raw_num,
                raw_confident=confident,
                number=raw_num,
                difficulty=stars,
                goal=goal,
                has_quiet_tag=has_tx,
                white=white,
                black=black,
                white_kings=w_kings,
                black_kings=b_kings,
                crop_img=board_gray,
            )
        )

    # Извлекаем текстовые строки ВНЕ диаграмм, не срезая строки решений под последним рядом
    lines_map: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}
    line_y: Dict[Tuple[int, int], float] = {}
    for wt in words:
        cx = (wt[0] + wt[2]) / 2.0 * scale
        cy = (wt[1] + wt[3]) / 2.0 * scale
        wtext = wt[4]
        skip_word = False
        for bx, by, bw, bh in raw_boxes:
            if (bx - 0.05 * bw) <= cx <= (bx + 1.05 * bw):
                # Внутри самой рамки доски — всегда шум
                if by <= cy <= (by + bh):
                    skip_word = True
                    break
                # Над доской и сразу под доской — режем только коды/номера диаграмм
                if ((by - 0.18 * bh) <= cy < by or (by + bh) < cy <= (by + 1.12 * bh)):
                    if DIAG_CODE_WORD_RE.fullmatch(wtext):
                        skip_word = True
                        break
        if skip_word:
            continue
        key = (wt[5], wt[6])
        lines_map.setdefault(key, []).append((wt[0], wtext))
        line_y[key] = min(line_y.get(key, wt[1]), wt[1])

    ordered_keys = sorted(lines_map.keys(), key=lambda k: (round(line_y[k] / 4.0), lines_map[k][0][0]))
    clean_lines = [
        " ".join(word for _, word in sorted(lines_map[k], key=lambda t: t[0]))
        for k in ordered_keys
    ]
    return detected, clean_lines


def assign_consensus_board_numbers(boards: List[DetectedBoard]) -> None:
    n = len(boards)
    if n == 0:
        return

    prev_offset: Optional[int] = None
    for i in range(n):
        votes: Counter[int] = Counter()
        lo, hi = max(0, i - 10), min(n, i + 11)
        for j in range(lo, hi):
            rn = boards[j].raw_number
            if rn is None or rn <= 0 or rn > 2800:
                continue
            off = rn - j
            weight = 3 if boards[j].raw_confident else 1
            votes[off] += weight

        if votes:
            max_w = max(votes.values())
            best_offs = [o for o, w in votes.items() if w == max_w]
            if prev_offset is not None and prev_offset in best_offs:
                chosen = prev_offset
            else:
                chosen = min(best_offs, key=lambda o: abs(o - prev_offset) if prev_offset is not None else 0)
            boards[i].number = i + chosen
            prev_offset = chosen
        elif prev_offset is not None:
            boards[i].number = i + prev_offset


# ============================================================================
# 5. ВАЛИДАЦИЯ С ВОССТАНОВЛЕНИЕМ ПРОПУЩЕННЫХ ВЫНУЖДЕННЫХ БОЁВ
# ============================================================================

@dataclass
class ValidationResult:
    ok: bool
    puzzle: Optional[dict] = None
    repairs: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _try_bridge_forced_captures(
    eng: RussianDraughtsEngine, tok: str, max_depth: int = 2
) -> Optional[List[Move]]:
    """
    Если текущий ход потерян OCR, но в позиции есть ровно ОДИН легальный ход
    и это обязательное взятие — проверяет, подойдёт ли текущий токен сразу после
    1 (или 2) таких единственных вынужденных боёв.
    """
    probe = eng.clone()
    bridge: List[Move] = []
    for _ in range(max_depth):
        legal = probe.legal_moves()
        if len(legal) != 1 or not legal[0].is_capture:
            return None
        forced = legal[0]
        bridge.append(forced)
        probe.apply(forced)
        next_mv, _ = match_move_with_repair(tok, probe.legal_moves())
        if next_mv is not None:
            return bridge
    return None


def _validate_single_candidate(
    number: int,
    white: List[str],
    black: List[str],
    white_kings: List[str],
    black_kings: List[str],
    cand: SolutionCandidate,
    section: str,
    difficulty: int,
    author: str,
    goal: str,
) -> ValidationResult:
    eng = RussianDraughtsEngine(white, black, white_kings, black_kings, turn="w")
    repairs: List[str] = []
    full_main: List[str] = []
    states_after_ply: Dict[int, RussianDraughtsEngine] = {0: eng.clone()}

    ply = 0
    tok_idx = 0
    bridged_total = 0

    while tok_idx < len(cand.main_tokens):
        tok = cand.main_tokens[tok_idx]
        legal = eng.legal_moves()
        mv, note = match_move_with_repair(tok, legal)

        if mv is None and bridged_total < 4:
            bridge = _try_bridge_forced_captures(eng, tok, max_depth=2)
            if bridge:
                for forced in bridge:
                    ply += 1
                    bridged_total += 1
                    repairs.append(
                        f"ply {ply}: Восстановлен пропущенный OCR вынужденный бой -> '{forced.to_full_notation()}'"
                    )
                    full_main.append(forced.to_full_notation())
                    eng.apply(forced)
                    states_after_ply[ply] = eng.clone()
                continue

        if mv is None:
            return ValidationResult(
                ok=False,
                repairs=repairs,
                error=f"Главная линия, полуход {ply + 1} ({'белые' if eng.turn == 'w' else 'чёрные'}): {note}",
            )

        ply += 1
        if note:
            repairs.append(f"ply {ply}: {note}")
        full_main.append(mv.to_full_notation())
        eng.apply(mv)
        states_after_ply[ply] = eng.clone()
        tok_idx += 1

    valid_branches: List[dict] = []
    for after_ply, br_tokens in cand.branches:
        if after_ply not in states_after_ply:
            continue
        br_eng = states_after_ply[after_ply].clone()
        br_full: List[str] = []
        br_ok = True
        for i, tok in enumerate(br_tokens):
            mv, note = match_move_with_repair(tok, br_eng.legal_moves())
            if mv is None:
                bridge = _try_bridge_forced_captures(br_eng, tok, max_depth=1)
                if bridge:
                    br_full.append(bridge[0].to_full_notation())
                    br_eng.apply(bridge[0])
                    mv, note = match_move_with_repair(tok, br_eng.legal_moves())
            if mv is None:
                br_ok = False
                repairs.append(f"Ветка после ply {after_ply} пропущена ({note})")
                break
            if note:
                repairs.append(f"branch@{after_ply}.{i + 1}: {note}")
            br_full.append(mv.to_full_notation())
            br_eng.apply(mv)
        if br_ok and br_full:
            valid_branches.append({"afterPly": after_ply, "moves": br_full})

    return ValidationResult(
        ok=True,
        puzzle={
            "id": f"tsukernik-{number}",
            "number": number,
            "section": section,
            "difficulty": difficulty,
            "author": author,
            "goal": goal,
            "white": sorted(white),
            "black": sorted(black),
            "whiteKings": sorted(white_kings),
            "blackKings": sorted(black_kings),
            "turn": "w",
            "mainLine": full_main,
            "branches": valid_branches,
        },
        repairs=repairs,
    )


def validate_puzzle_candidates(
    number: int,
    white: List[str],
    black: List[str],
    white_kings: List[str],
    black_kings: List[str],
    candidates: List[SolutionCandidate],
    section: str = "Антология Цукерника",
    difficulty: int = 1,
    author: str = "",
    goal: str = "win",
) -> ValidationResult:
    if not white and not white_kings:
        return ValidationResult(ok=False, error="На доске нет белых шашек")
    if not black and not black_kings:
        return ValidationResult(ok=False, error="На доске нет чёрных шашек")

    first_err: Optional[ValidationResult] = None
    for cand in candidates:
        res = _validate_single_candidate(
            number, white, black, white_kings, black_kings,
            cand, section, difficulty, author, goal,
        )
        if res.ok:
            return res
        if first_err is None:
            first_err = res
    return first_err or ValidationResult(ok=False, error="Пустое решение")


# ============================================================================
# 6. CLI-ЗАПУСК
# ============================================================================

def run_pdf_pipeline(
    pdf_path: Path,
    start_page: int,
    end_page: int,
    out_verified: Path,
    out_review: Path,
    debug_dir: Path,
    template_dir: Optional[Path],
) -> None:
    doc = fitz.open(pdf_path)
    templates = load_templates(template_dir)
    all_boards: List[DetectedBoard] = []
    all_lines: List[str] = []
    current_section = "Антология Цукерника"
    section_by_page: Dict[int, str] = {}

    for pno in range(start_page, min(end_page, len(doc)) + 1):
        page = doc[pno - 1]
        boards, clean_lines = detect_boards_and_text_lines(page, pno, templates)
        all_boards.extend(boards)
        all_lines.extend(clean_lines)

        page_joined = " ".join(clean_lines)
        m_sec = re.search(r"ФИНАЛЬНЫЙ УДАР ШАШКОЙ\s+([a-hа-я0-9]{2})", page_joined, re.IGNORECASE)
        if m_sec:
            current_section = f"Финальный удар шашкой {m_sec.group(1)}"
        section_by_page[pno] = current_section

    assign_consensus_board_numbers(all_boards)
    target_nums = {b.number for b in all_boards if b.number is not None}
    solutions = extract_solutions_from_lines(all_lines, target_nums)

    verified: List[dict] = []
    needs_review: List[dict] = []
    debug_dir.mkdir(parents=True, exist_ok=True)

    for b in all_boards:
        if b.number is None or b.number not in solutions:
            crop_file = debug_dir / f"page{b.page}_num_{b.number or 'unknown'}.png"
            if b.crop_img is not None:
                cv2.imwrite(str(crop_file), b.crop_img)
            needs_review.append({
                "number": b.number,
                "page": b.page,
                "error": "Не найдено решение в OCR-тексте диапазона",
                "crop": str(crop_file),
                "white": b.white,
                "black": b.black,
                "whiteKings": b.white_kings,
                "blackKings": b.black_kings,
            })
            continue

        sol = solutions[b.number]
        res = validate_puzzle_candidates(
            number=b.number,
            white=b.white,
            black=b.black,
            white_kings=b.white_kings,
            black_kings=b.black_kings,
            candidates=sol.candidates,
            section=section_by_page.get(b.page, "Антология Цукерника"),
            difficulty=b.difficulty,
            author=sol.author,
            goal=b.goal,
        )

        if res.ok and res.puzzle is not None:
            verified.append(res.puzzle)
            if res.repairs:
                print(f"[OK + REPAIR] №{b.number} (стр. {b.page}): {len(res.repairs)} испр.")
            else:
                print(f"[OK]          №{b.number} (стр. {b.page})")
        else:
            crop_file = debug_dir / f"puzzle_{b.number}.png"
            if b.crop_img is not None:
                cv2.imwrite(str(crop_file), b.crop_img)
            needs_review.append({
                "number": b.number,
                "page": b.page,
                "error": res.error,
                "repairs_before_fail": res.repairs,
                "raw_solution": sol.raw_text,
                "parsed_tokens": sol.candidates[0].main_tokens if sol.candidates else [],
                "crop": str(crop_file),
                "white": b.white,
                "black": b.black,
                "whiteKings": b.white_kings,
                "blackKings": b.black_kings,
            })
            print(f"[FAIL]        №{b.number} (стр. {b.page}): {res.error}")

    out_verified.write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    out_review.write_text(json.dumps(needs_review, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Диаграмм найдено:       {len(all_boards)}")
    print(f"Решений распарсено:     {len(solutions)}")
    print(f"Валидно (готово в UI):  {len(verified)} -> {out_verified}")
    print(f"Требуют проверки:       {len(needs_review)} -> {out_review}")
    print(f"Кропы ошибок сохранены: {debug_dir}/")


def run_json_validation(json_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    ok_count = 0
    for item in data:
        branches = [(br["afterPly"], br["moves"]) for br in item.get("branches", [])]
        cand = SolutionCandidate(main_tokens=item.get("mainLine", []), branches=branches)
        res = validate_puzzle_candidates(
            number=item.get("number", 0),
            white=item.get("white", []),
            black=item.get("black", []),
            white_kings=item.get("whiteKings", []),
            black_kings=item.get("blackKings", []),
            candidates=[cand],
            section=item.get("section", ""),
            difficulty=item.get("difficulty", 1),
            author=item.get("author", ""),
            goal=item.get("goal", "win"),
        )
        if res.ok:
            ok_count += 1
        else:
            print(f"[FAIL] №{item.get('number')} ({item.get('id')}): {res.error}")
    print(f"\nИтог валидации {json_path.name}: {ok_count}/{len(data)} задач полностью корректны.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оцифровка и валидация «Антологии» Цукерника")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pdf = sub.add_parser("pdf", help="Оцифровать и провалидировать диапазон страниц PDF")
    p_pdf.add_argument("pdf_file", type=Path)
    p_pdf.add_argument("--pages", default="12-29", help="Диапазон страниц, например 12-29")
    p_pdf.add_argument("--out", type=Path, default=Path("tsukernik_verified.json"))
    p_pdf.add_argument("--review", type=Path, default=Path("tsukernik_needs_review.json"))
    p_pdf.add_argument("--debug-dir", type=Path, default=Path("debug_crops"))
    p_pdf.add_argument("--templates", type=Path, default=None)

    p_val = sub.add_parser("validate", help="Проверить существующий JSON-файл задач")
    p_val.add_argument("json_file", type=Path)

    args = parser.parse_args()
    if args.cmd == "pdf":
        s, e = map(int, args.pages.split("-"))
        run_pdf_pipeline(args.pdf_file, s, e, args.out, args.review, args.debug_dir, args.templates)
    else:
        run_json_validation(args.json_file)


if __name__ == "__main__":
    main()