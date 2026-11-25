#!/usr/bin/env python3
"""
Быстрый тактический движок русских шашек 8x8 (FastBoard int[64] + инкрементальный Zobrist +
Aspiration Windows + PVS + LMR + Quiescence + продление форсированных вариантов + автопоиск комбинаций).

Совместим по API с tsukernik_validator.RussianDraughtsEngine.
"""

from __future__ import annotations
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tsukernik_validator import Move, RussianDraughtsEngine, FILES, is_dark, rc_to_sq, sq_to_rc

INF = 100_000
WIN_SCORE = 50_000

EXACT = 0
LOWER = 1
UPPER = 2

# Коды фигур на 64-клеточной доске (r*8 + c)
EMPTY = 0
W_MAN = 1
W_KING = 2
B_MAN = 3
B_KING = 4

# Таблицы Zobrist (64 клетки x 5 кодов; индекс 0..4)
_rng = random.Random(20261125)
PIECE_TO_IDX = {"w": 0, "W": 1, "b": 2, "B": 3}
ZOBRIST_SQ = [[_rng.getrandbits(64) for _ in range(4)] for _ in range(64)]
# Индексировано напрямую кодом фигуры 1..4 (для FastBoard)
_Z_FAST = [[0] * 5 for _ in range(64)]
for _sq in range(64):
    for _k in range(4):
        _Z_FAST[_sq][_k + 1] = ZOBRIST_SQ[_sq][_k]
ZOBRIST_TURN = _rng.getrandbits(64)

# Веса позиционной оценки (русские шашки 8x8)
CENTER_SQ = {
    sq_to_rc("c3"), sq_to_rc("e3"), sq_to_rc("d4"), sq_to_rc("f4"),
    sq_to_rc("c5"), sq_to_rc("e5"), sq_to_rc("d6"), sq_to_rc("f6"),
}
BACK_GUARD_W = {sq_to_rc("a1"), sq_to_rc("c1"), sq_to_rc("e1"), sq_to_rc("g1")}
BACK_GUARD_B = {sq_to_rc("b8"), sq_to_rc("d8"), sq_to_rc("f8"), sq_to_rc("h8")}
RANK_BONUS = (0, 4, 9, 16, 26, 40, 62, 0)

# Предвычисленная таблица весов (белые +, чёрные -) для FastBoard
_FAST_PSQT = [[0] * 64 for _ in range(5)]
_BACK_W_IDX = tuple(r * 8 + c for (r, c) in BACK_GUARD_W)
_BACK_B_IDX = tuple(r * 8 + c for (r, c) in BACK_GUARD_B)
for _r in range(8):
    for _c in range(8):
        if (_r + _c) & 1:
            continue
        _s = _r * 8 + _c
        _center = 1 if (_r, _c) in CENTER_SQ else 0
        _edge = 1 if (_c == 0 or _c == 7) else 0
        _main_diag = 1 if _r == _c else 0
        _FAST_PSQT[W_MAN][_s] = 100 + RANK_BONUS[_r] + 12 * _center - 5 * _edge
        _FAST_PSQT[W_KING][_s] = 360 + 18 * _center + 14 * _main_diag
        _FAST_PSQT[B_MAN][_s] = -(100 + RANK_BONUS[7 - _r] + 12 * _center - 5 * _edge)
        _FAST_PSQT[B_KING][_s] = -(360 + 18 * _center + 14 * _main_diag)

# Предвычисленные геометрические таблицы для 64 клеток
_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_DARK_SQUARES = tuple(r * 8 + c for r in range(8) for c in range(8) if (r + c) % 2 == 0)

# _RAYS[sq][dir_idx] -> кортеж клеток от sq в направлении dir_idx
_RAYS: Tuple[Tuple[Tuple[int, ...], ...], ...] = tuple(
    tuple(
        tuple(
            (r + k * dr) * 8 + (c + k * dc)
            for k in range(1, 8)
            if 0 <= r + k * dr < 8 and 0 <= c + k * dc < 8
        )
        for dr, dc in _DIRS
    )
    for sq in range(64)
    for r, c in ((sq >> 3, sq & 7),)
)

# _JUMP_ADJ[sq] -> кортеж (mid_sq, land_sq) для шашек
_JUMP_ADJ: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple(
    tuple(
        ((r + dr) * 8 + (c + dc), (r + 2 * dr) * 8 + (c + 2 * dc))
        for dr, dc in _DIRS
        if 0 <= r + 2 * dr < 8 and 0 <= c + 2 * dc < 8
    )
    for sq in range(64)
    for r, c in ((sq >> 3, sq & 7),)
)

# Тихие шаги простых шашек: _MAN_STEPS[color_int (0=w, 1=b)][sq] -> кортеж (to_sq, promotes)
_MAN_STEPS = [[() for _ in range(64)], [() for _ in range(64)]]
for _sq in _DARK_SQUARES:
    _r, _c = _sq >> 3, _sq & 7
    _w_list = []
    if _r + 1 < 8:
        for _dc in (-1, 1):
            _nc = _c + _dc
            if 0 <= _nc < 8:
                _w_list.append(((_r + 1) * 8 + _nc, _r + 1 == 7))
    _MAN_STEPS[0][_sq] = tuple(_w_list)
    _b_list = []
    if _r - 1 >= 0:
        for _dc in (-1, 1):
            _nc = _c + _dc
            if 0 <= _nc < 8:
                _b_list.append(((_r - 1) * 8 + _nc, _r - 1 == 0))
    _MAN_STEPS[1][_sq] = tuple(_b_list)


def compute_zobrist(eng: RussianDraughtsEngine) -> int:
    h = ZOBRIST_TURN if eng.turn == "b" else 0
    for (r, c), p in eng.board.items():
        h ^= ZOBRIST_SQ[r * 8 + c][PIECE_TO_IDX[p]]
    return h


def evaluate_position(eng: RussianDraughtsEngine) -> int:
    """Оценка позиции с точки зрения стороны, чей сейчас ход."""
    score = 0
    w_cnt = 0
    b_cnt = 0
    w_kings = 0
    b_kings = 0

    for (r, c), p in eng.board.items():
        if p == "w":
            w_cnt += 1
            score += _FAST_PSQT[W_MAN][r * 8 + c]
        elif p == "W":
            w_kings += 1
            score += _FAST_PSQT[W_KING][r * 8 + c]
        elif p == "b":
            b_cnt += 1
            score += _FAST_PSQT[B_MAN][r * 8 + c]
        else:
            b_kings += 1
            score += _FAST_PSQT[B_KING][r * 8 + c]

    if b_kings == 0:
        for sq in BACK_GUARD_W:
            if eng.board.get(sq) == "w":
                score += 7
    if w_kings == 0:
        for sq in BACK_GUARD_B:
            if eng.board.get(sq) == "b":
                score -= 7

    if w_cnt + w_kings > b_cnt + b_kings and b_cnt + b_kings <= 3:
        score += 15 * (w_kings - b_kings)
    elif b_cnt + b_kings > w_cnt + w_kings and w_cnt + w_kings <= 3:
        score -= 15 * (b_kings - w_kings)

    return score if eng.turn == "w" else -score


class _FastBoard:
    """Быстрая плоская 64-клеточная доска с инкрементальным Zobrist и eval для поиска."""

    __slots__ = ("cells", "turn", "z_hash", "psqt", "counts")

    def __init__(self, eng: RussianDraughtsEngine):
        self.cells = [0] * 64
        self.turn = 0 if eng.turn == "w" else 1
        self.z_hash = ZOBRIST_TURN if self.turn == 1 else 0
        self.psqt = 0
        self.counts = [0, 0, 0, 0, 0]  # _, W_MAN, W_KING, B_MAN, B_KING

        cmap = {"w": W_MAN, "W": W_KING, "b": B_MAN, "B": B_KING}
        for (r, c), p in eng.board.items():
            sq = r * 8 + c
            code = cmap[p]
            self.cells[sq] = code
            self.z_hash ^= _Z_FAST[sq][code]
            self.psqt += _FAST_PSQT[code][sq]
            self.counts[code] += 1

    def evaluate(self) -> int:
        sc = self.psqt
        cells = self.cells
        w_m, w_k, b_m, b_k = self.counts[1], self.counts[2], self.counts[3], self.counts[4]
        if b_k == 0:
            if cells[0] == W_MAN: sc += 7
            if cells[2] == W_MAN: sc += 7
            if cells[4] == W_MAN: sc += 7
            if cells[6] == W_MAN: sc += 7
        if w_k == 0:
            if cells[57] == B_MAN: sc -= 7
            if cells[59] == B_MAN: sc -= 7
            if cells[61] == B_MAN: sc -= 7
            if cells[63] == B_MAN: sc -= 7
        w_tot = w_m + w_k
        b_tot = b_m + b_k
        if w_tot > b_tot and b_tot <= 3:
            sc += 15 * (w_k - b_k)
        elif b_tot > w_tot and w_tot <= 3:
            sc -= 15 * (b_k - w_k)
        return sc if self.turn == 0 else -sc

    def _find_captures(
        self,
        sq: int,
        turn: int,
        is_king: bool,
        captured_mask: int,
    ) -> List[Tuple[int, Tuple[int, ...], bool]]:
        """Возвращает [(end_sq, cap_squares_tuple, final_king)] для рекурсивных взятий."""
        cells = self.cells
        enemy_lo = 3 if turn == 0 else 1
        enemy_hi = 4 if turn == 0 else 2
        promo_rank = 7 if turn == 0 else 0
        results: List[Tuple[int, Tuple[int, ...], bool]] = []

        if not is_king:
            for mr, lr in _JUMP_ADJ[sq]:
                target = cells[mr]
                if target < enemy_lo or target > enemy_hi:
                    continue
                bit = 1 << mr
                if (captured_mask & bit) or cells[lr] != 0:
                    continue
                next_mask = captured_mask | bit
                next_king = (lr >> 3) == promo_rank
                sub = self._find_captures(lr, turn, next_king, next_mask)
                if sub:
                    for end_sq, tail_caps, f_king in sub:
                        results.append((end_sq, (mr,) + tail_caps, f_king))
                else:
                    results.append((lr, (mr,), next_king))
        else:
            for ray in _RAYS[sq]:
                # Ищем первого врага на луче
                idx = 0
                n_ray = len(ray)
                while idx < n_ray and cells[ray[idx]] == 0:
                    idx += 1
                if idx >= n_ray:
                    continue
                sr = ray[idx]
                target = cells[sr]
                if target < enemy_lo or target > enemy_hi or (captured_mask & (1 << sr)):
                    continue

                # Собираем пустые поля за побитой шашкой
                land_idx = idx + 1
                landings: List[int] = []
                while land_idx < n_ray and cells[ray[land_idx]] == 0:
                    landings.append(ray[land_idx])
                    land_idx += 1
                if not landings:
                    continue

                next_mask = captured_mask | (1 << sr)
                continuing: List[Tuple[int, Tuple[int, ...], bool]] = []
                for land in landings:
                    sub = self._find_captures(land, turn, True, next_mask)
                    for end_sq, tail_caps, f_king in sub:
                        continuing.append((end_sq, (sr,) + tail_caps, f_king))
                if continuing:
                    results.extend(continuing)
                else:
                    for land in landings:
                        results.append((land, (sr,), True))
        return results

    def legal_moves(self) -> List[Tuple[int, int, Tuple[int, ...], bool, int]]:
        """
        Возвращает список ходов: (start_sq, end_sq, caps, promotes, move_key)
        move_key = (start_sq << 12) | (end_sq << 6) | len(caps)
        """
        cells = self.cells
        turn = self.turn
        man_code = W_MAN if turn == 0 else B_MAN
        king_code = W_KING if turn == 0 else B_KING

        captures: List[Tuple[int, int, Tuple[int, ...], bool, int]] = []
        seen_cap: Set[Tuple[int, int, Tuple[int, ...]]] = set()

        for sq in _DARK_SQUARES:
            p = cells[sq]
            if p != man_code and p != king_code:
                continue
            is_king = p == king_code
            # Снимаем бьющую фигуру на время поиска цепочки (как в RussianDraughtsEngine)
            cells[sq] = 0
            chains = self._find_captures(sq, turn, is_king, 0)
            cells[sq] = p
            for end_sq, caps, final_king in chains:
                k = (sq, end_sq, caps)
                if k in seen_cap:
                    continue
                seen_cap.add(k)
                promotes = (not is_king) and final_king
                mkey = (sq << 12) | (end_sq << 6) | len(caps)
                captures.append((sq, end_sq, caps, promotes, mkey))

        if captures:
            return captures

        quiet: List[Tuple[int, int, Tuple[int, ...], bool, int]] = []
        man_steps = _MAN_STEPS[turn]
        for sq in _DARK_SQUARES:
            p = cells[sq]
            if p == man_code:
                for to_sq, prom in man_steps[sq]:
                    if cells[to_sq] == 0:
                        mkey = (sq << 12) | (to_sq << 6)
                        quiet.append((sq, to_sq, (), prom, mkey))
            elif p == king_code:
                for ray in _RAYS[sq]:
                    for to_sq in ray:
                        if cells[to_sq] != 0:
                            break
                        mkey = (sq << 12) | (to_sq << 6)
                        quiet.append((sq, to_sq, (), False, mkey))
        return quiet

    def push(self, mv: Tuple[int, int, Tuple[int, ...], bool, int]):
        start_sq, end_sq, caps, promotes, _ = mv
        cells = self.cells
        piece = cells[start_sq]
        old_hash = self.z_hash
        old_psqt = self.psqt

        # Снимаем с start_sq
        cells[start_sq] = 0
        self.z_hash ^= _Z_FAST[start_sq][piece]
        self.psqt -= _FAST_PSQT[piece][start_sq]

        # Удаляем побитые
        cap_pieces = []
        if caps:
            counts = self.counts
            z_fast = _Z_FAST
            fpsqt = _FAST_PSQT
            zh = self.z_hash
            ps = self.psqt
            for csq in caps:
                cp = cells[csq]
                cap_pieces.append((csq, cp))
                cells[csq] = 0
                counts[cp] -= 1
                zh ^= z_fast[csq][cp]
                ps -= fpsqt[cp][csq]
            self.z_hash = zh
            self.psqt = ps

        new_piece = piece
        if promotes:
            new_piece = W_KING if piece == W_MAN else B_KING
            self.counts[piece] -= 1
            self.counts[new_piece] += 1

        cells[end_sq] = new_piece
        self.z_hash ^= _Z_FAST[end_sq][new_piece] ^ ZOBRIST_TURN
        self.psqt += _FAST_PSQT[new_piece][end_sq]
        self.turn ^= 1
        return (start_sq, end_sq, piece, new_piece, cap_pieces, old_hash, old_psqt)

    def pop(self, undo) -> None:
        start_sq, end_sq, piece, new_piece, cap_pieces, old_hash, old_psqt = undo
        cells = self.cells
        counts = self.counts
        cells[end_sq] = 0
        cells[start_sq] = piece
        if piece != new_piece:
            counts[new_piece] -= 1
            counts[piece] += 1
        for csq, cp in cap_pieces:
            cells[csq] = cp
            counts[cp] += 1
        self.turn ^= 1
        self.z_hash = old_hash
        self.psqt = old_psqt

    def move_to_engine_move(self, mv: Tuple[int, int, Tuple[int, ...], bool, int], eng: RussianDraughtsEngine) -> Optional[Move]:
        s0, e0, caps0, prom0, _ = mv
        r0, c0 = s0 >> 3, s0 & 7
        r1, c1 = e0 >> 3, e0 & 7
        cap_set = {(cs >> 3, cs & 7) for cs in caps0}
        for emv in eng.legal_moves():
            if emv.path[0] == (r0, c0) and emv.path[-1] == (r1, c1) and emv.promotes == prom0:
                if not caps0 and not emv.captured:
                    return emv
                if len(emv.captured) == len(caps0) and set(emv.captured) == cap_set:
                    return emv
        return None


@dataclass
class SearchStats:
    nodes: int = 0
    tt_hits: int = 0
    depth_reached: int = 0
    elapsed_ms: float = 0.0


class TacticalDraughtsSolver:
    def __init__(self, tt_size: int = 1 << 18):
        self.tt_mask = tt_size - 1
        # TT: tt_idx -> (full_hash, depth, flag, score, best_move_key)
        self.tt: Dict[int, Tuple[int, int, int, int, int]] = {}
        self.killers: Dict[int, List[int]] = {}
        self.history: Dict[int, int] = {}
        self.nodes = 0
        self.tt_hits = 0
        self.stop_time = 0.0
        self.aborted = False

    def clear(self) -> None:
        self.tt.clear()
        self.killers.clear()
        self.history.clear()

    def _order_moves(
        self,
        moves: List[Tuple[int, int, Tuple[int, ...], bool, int]],
        tt_mkey: int,
        ply: int,
    ) -> List[Tuple[int, int, Tuple[int, ...], bool, int]]:
        if len(moves) <= 1:
            return moves
        k_list = self.killers.get(ply, ())
        hist = self.history

        def key_fn(mv: Tuple[int, int, Tuple[int, ...], bool, int]) -> int:
            mkey = mv[4]
            if mkey == tt_mkey:
                return 1_000_000
            sc = 0
            if mv[2]:  # caps
                sc += 50_000 + len(mv[2]) * 1_000
            if mv[3]:  # promotes
                sc += 20_000
            if mkey in k_list:
                sc += 10_000
            hval = hist.get(mkey, 0)
            sc += hval if hval < 5_000 else 5_000
            return sc

        moves.sort(key=key_fn, reverse=True)
        return moves

    def _quiescence(
        self,
        fb: _FastBoard,
        alpha: int,
        beta: int,
        ply: int,
        q_depth: int = 0,
    ) -> int:
        self.nodes += 1
        if (self.nodes & 4095) == 0 and time.perf_counter() >= self.stop_time:
            self.aborted = True
            return 0

        legal = fb.legal_moves()
        if not legal:
            return -WIN_SCORE + ply

        if not legal[0][2] or q_depth >= 8:
            return fb.evaluate()

        ordered = self._order_moves(legal, -1, ply)
        best = -INF
        for mv in ordered:
            undo = fb.push(mv)
            val = -self._quiescence(fb, -beta, -alpha, ply + 1, q_depth + 1)
            fb.pop(undo)
            if self.aborted:
                return 0
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best

    def _alphabeta(
        self,
        fb: _FastBoard,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        rep_stack: List[int],
    ) -> int:
        self.nodes += 1
        if (self.nodes & 4095) == 0 and time.perf_counter() >= self.stop_time:
            self.aborted = True
            return 0

        z_hash = fb.z_hash
        if ply > 0 and z_hash in rep_stack:
            return 0

        legal = fb.legal_moves()
        if not legal:
            return -WIN_SCORE + ply

        # Форсированное продление (единственный ответ или обязательный удар)
        ext = 0
        if len(legal) == 1 and ply < 22:
            ext = 1
        elif legal[0][2] and depth <= 2 and ply < 22:
            ext = 1
        eff_depth = depth + ext

        if eff_depth <= 0:
            return self._quiescence(fb, alpha, beta, ply)

        orig_alpha = alpha
        tt_idx = z_hash & self.tt_mask
        tt_entry = self.tt.get(tt_idx)
        tt_mkey = -1
        if tt_entry is not None and tt_entry[0] == z_hash:
            _, t_depth, t_flag, t_score, tt_mkey = tt_entry
            if t_depth >= eff_depth:
                self.tt_hits += 1
                if t_flag == EXACT:
                    return t_score
                if t_flag == LOWER and t_score > alpha:
                    alpha = t_score
                elif t_flag == UPPER and t_score < beta:
                    beta = t_score
                if alpha >= beta:
                    return t_score

        ordered = self._order_moves(legal, tt_mkey, ply)
        best_score = -INF
        best_mkey = -1
        k_list = self.killers.get(ply, ())
        rep_stack.append(z_hash)

        try:
            for idx, mv in enumerate(ordered):
                undo = fb.push(mv)
                is_quiet = (not mv[2]) and (not mv[3])
                mkey = mv[4]

                if idx == 0:
                    score = -self._alphabeta(fb, eff_depth - 1, -beta, -alpha, ply + 1, rep_stack)
                else:
                    # LMR для поздних тихих ходов
                    reduce = 0
                    if eff_depth >= 3 and idx >= 4 and is_quiet and (mkey not in k_list):
                        reduce = 1 if idx < 8 else 2
                    search_d = max(0, eff_depth - 1 - reduce)
                    score = -self._alphabeta(fb, search_d, -alpha - 1, -alpha, ply + 1, rep_stack)
                    if not self.aborted and reduce > 0 and score > alpha:
                        score = -self._alphabeta(fb, eff_depth - 1, -alpha - 1, -alpha, ply + 1, rep_stack)
                    if not self.aborted and alpha < score < beta:
                        score = -self._alphabeta(fb, eff_depth - 1, -beta, -alpha, ply + 1, rep_stack)

                fb.pop(undo)
                if self.aborted:
                    return 0

                if score > best_score:
                    best_score = score
                    best_mkey = mkey
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    if not mv[2] and best_mkey >= 0:
                        ks = self.killers.setdefault(ply, [])
                        if best_mkey not in ks:
                            ks.insert(0, best_mkey)
                            if len(ks) > 2:
                                ks.pop()
                        self.history[best_mkey] = self.history.get(best_mkey, 0) + eff_depth * eff_depth
                    break
        finally:
            rep_stack.pop()

        flag = EXACT
        if best_score <= orig_alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        self.tt[tt_idx] = (z_hash, eff_depth, flag, best_score, best_mkey)
        return best_score

    def _extract_pv(self, eng: RussianDraughtsEngine, max_len: int = 16) -> Tuple[Optional[Move], List[str]]:
        pv: List[str] = []
        first_mv: Optional[Move] = None
        probe_eng = eng.clone()
        fb = _FastBoard(probe_eng)
        seen = set()
        for step in range(max_len):
            zh = fb.z_hash
            if zh in seen:
                break
            seen.add(zh)
            entry = self.tt.get(zh & self.tt_mask)
            if entry is None or entry[0] != zh or entry[4] < 0:
                break
            mkey = entry[4]
            f_mv = next((m for m in fb.legal_moves() if m[4] == mkey), None)
            if f_mv is None:
                break
            emv = fb.move_to_engine_move(f_mv, probe_eng)
            if emv is None:
                break
            if step == 0:
                first_mv = emv
            pv.append(emv.to_full_notation())
            probe_eng.apply(emv)
            fb.push(f_mv)
        return first_mv, pv

    def search(
        self,
        eng: RussianDraughtsEngine,
        max_depth: int = 10,
        time_limit_sec: float = 1.5,
    ) -> Tuple[Optional[Move], int, List[str], SearchStats]:
        t0 = time.perf_counter()
        self.stop_time = t0 + time_limit_sec
        self.nodes = 0
        self.tt_hits = 0
        self.aborted = False

        legal = eng.legal_moves()
        if not legal:
            return None, -WIN_SCORE, [], SearchStats()
        if len(legal) == 1 and max_depth <= 1:
            mv = legal[0]
            nxt = eng.clone()
            nxt.apply(mv)
            return mv, -evaluate_position(nxt), [mv.to_full_notation()], SearchStats(nodes=1, depth_reached=1)

        fb = _FastBoard(eng)
        best_mv: Optional[Move] = legal[0]
        best_score = 0
        best_pv: List[str] = [legal[0].to_full_notation()]
        depth_reached = 0
        rep_stack: List[int] = []

        for depth in range(1, max_depth + 1):
            if depth >= 4:
                window = 55
                a = max(-INF, best_score - window)
                b = min(INF, best_score + window)
                score = self._alphabeta(fb, depth, a, b, 0, rep_stack)
                if not self.aborted and (score <= a or score >= b):
                    score = self._alphabeta(fb, depth, -INF, INF, 0, rep_stack)
            else:
                score = self._alphabeta(fb, depth, -INF, INF, 0, rep_stack)

            if self.aborted:
                break
            depth_reached = depth
            best_score = score
            root_mv, pv = self._extract_pv(eng, max_len=16)
            if root_mv is not None and pv:
                best_mv = root_mv
                best_pv = pv
            if abs(score) >= WIN_SCORE - 100:
                break

        stats = SearchStats(
            nodes=self.nodes,
            tt_hits=self.tt_hits,
            depth_reached=depth_reached,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return best_mv, best_score, best_pv, stats

    def find_combination(
        self,
        eng: RussianDraughtsEngine,
        max_depth: int = 10,
        time_limit_sec: float = 2.0,
        min_advantage: int = 160,
        min_gap_over_second: int = 110,
    ) -> dict:
        """
        Автопоиск комбинации: оценивает ВСЕ легальные первые ходы, проверяя,
        есть ли явный выигрыш/комбинация с единственным сильнейшим первым ходом.
        """
        t0 = time.perf_counter()
        legal = eng.legal_moves()
        if not legal:
            return {"found": False, "reason": "Нет легальных ходов"}
        if len(legal) == 1:
            mv, sc, pv, st = self.search(eng, max_depth=max_depth, time_limit_sec=time_limit_sec)
            return {
                "found": sc >= min_advantage,
                "best_move": pv[0] if pv else legal[0].to_full_notation(),
                "score": sc,
                "gap": 9999,
                "pv": pv,
                "depth": st.depth_reached,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            }

        # Стадия 1: быстрый скрининг всех легальных ходов на глубину 5 (без очистки TT!)
        self.clear()
        stage1_results: List[Tuple[int, Move, List[str], int, RussianDraughtsEngine]] = []
        screen_time = min(0.04, (time_limit_sec * 0.28) / len(legal))
        for mv in legal:
            nxt = eng.clone()
            nxt.apply(mv)
            _, child_sc, child_pv, st = self.search(nxt, max_depth=min(5, max(1, max_depth - 1)), time_limit_sec=screen_time)
            sc = -child_sc
            pv = [mv.to_full_notation()] + child_pv
            # Небольшой приоритет жертвам (если после хода у соперника вынужденное взятие / мало ходов)
            opp_legal = nxt.legal_moves()
            forced_bonus = 25 if len(opp_legal) == 1 else (10 if opp_legal and opp_legal[0].is_capture else 0)
            stage1_results.append((sc + forced_bonus, sc, mv, pv, st.depth_reached + 1, nxt))

        stage1_results.sort(key=lambda x: x[0], reverse=True)

        # Стадия 2: глубокий пересчёт лучших кандидатов на оставшийся бюджет времени
        rem_time = max(0.15, time_limit_sec - (time.perf_counter() - t0))
        top_k = min(len(stage1_results), 4 if len(stage1_results) >= 4 else len(stage1_results))
        per_top_time = rem_time / top_k

        final_results: List[Tuple[int, Move, List[str], int]] = []
        for idx, (_, sc1, mv, pv1, d1, nxt) in enumerate(stage1_results):
            if idx < top_k:
                _, child_sc, child_pv, st = self.search(
                    nxt, max_depth=max(1, max_depth - 1), time_limit_sec=per_top_time
                )
                sc = -child_sc
                pv = [mv.to_full_notation()] + child_pv
                final_results.append((sc, mv, pv, st.depth_reached + 1))
            else:
                final_results.append((sc1, mv, pv1, d1))

        final_results.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_mv, best_pv, best_d = final_results[0]
        second_sc = final_results[1][0]
        gap = best_sc - second_sc

        return {
            "found": best_sc >= min_advantage and gap >= min_gap_over_second,
            "best_move": best_mv.to_full_notation(),
            "second_move": final_results[1][1].to_full_notation(),
            "score": best_sc,
            "second_score": second_sc,
            "gap": gap,
            "pv": best_pv,
            "depth": best_d,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }


def benchmark_on_tsukernik(limit: int = 40) -> None:
    gold_path = Path(__file__).resolve().parent / "tsukernik_gold.json"
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    sample = [p for p in data if len(p.get("mainLine", [])) <= 8][:limit]

    solver = TacticalDraughtsSolver()
    matched_first = 0
    total_ms = 0.0
    total_nodes = 0
    total_depth = 0

    for p in sample:
        solver.clear()
        eng = RussianDraughtsEngine(
            p["white"], p["black"], p.get("whiteKings", []), p.get("blackKings", []), turn=p.get("turn", "w")
        )
        mv, sc, pv, st = solver.search(eng, max_depth=10, time_limit_sec=0.6)
        total_ms += st.elapsed_ms
        total_nodes += st.nodes
        total_depth += st.depth_reached
        expected_first = p["mainLine"][0]
        if mv and mv.to_full_notation() == expected_first:
            matched_first += 1

    print(
        f"Benchmark ({len(sample)} Tsoukernik puzzles <=8 plies, 0.6s limit): "
        f"first-move accuracy = {matched_first}/{len(sample)} ({100*matched_first/len(sample):.1f}%), "
        f"avg time = {total_ms/len(sample):.1f} ms, avg nodes = {total_nodes//len(sample)}, avg depth = {total_depth/len(sample):.1f}"
    )


if __name__ == "__main__":
    benchmark_on_tsukernik(40)
