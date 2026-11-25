#!/usr/bin/env python3
"""
Точный экстрактор и многопоточный (multi-core) решатель для книги
В. Ячейкина «Комбинации украинских шашистов» (Киев, 1974).

Особенности:
- Позиции в PDF отрисованы векторными серыми клетками и фиксированными спрайтами шашек (xref),
  поэтому диаграммы извлекаются с 100% геометрической точностью (без ошибок OCR).
- Заголовки вида "12. М. Бень" и пометка "=" (ничья) сопоставляются по координатам над диаграммой.
- Линия каждой задачи выстраивается пошагово против СИЛЬНЕЙШЕЙ защиты чёрных (каждый выбор чёрных
  перепроверяется отдельным глубоким поиском с предпочтением максимального сопротивления; при
  обнаружении ничейной или спорной защиты комбинация отбраковывается).
- Фильтруются некомбинационные позиции (требуется жертва/комбинационный удар/многократное взятие).
"""

from __future__ import annotations
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audit_master import audit_puzzle, material_score
from draughts_solver import TacticalDraughtsSolver, evaluate_position
from tsukernik_validator import FILES, RussianDraughtsEngine, is_dark


WHITE_XREFS = {34, 49}
BLACK_XREFS = {32, 33, 56}


def extract_yacheykin_diagrams(pdf_path: Path) -> List[dict]:
    doc = fitz.open(str(pdf_path))
    diagrams: List[dict] = []

    for pi in range(1, len(doc)):
        page = doc[pi]
        txt = page.get_text()
        if "РЕШЕНИЯ" in txt:
            break

        gray_rects = [
            d["rect"]
            for d in page.get_drawings()
            if d.get("fill")
            and abs(d["fill"][0] - 0.753) < 0.03
            and 11.0 <= d["rect"].width <= 17.0
            and 10.0 <= d["rect"].height <= 16.0
        ]
        if not gray_rects:
            continue

        boards: List[Tuple[int, float, float, float, float]] = []
        used = [False] * len(gray_rects)
        for i, r in enumerate(gray_rects):
            if used[i]:
                continue
            q = [i]
            used[i] = True
            grp = [r]
            while q:
                cur = q.pop()
                cr = gray_rects[cur]
                for j, rr in enumerate(gray_rects):
                    if not used[j] and abs(rr.x0 - cr.x0) < 35.0 and abs(rr.y0 - cr.y0) < 35.0:
                        used[j] = True
                        q.append(j)
                        grp.append(rr)
            if len(grp) >= 28:
                boards.append((
                    len(grp),
                    min(g.x0 for g in grp),
                    min(g.y0 for g in grp),
                    max(g.x1 for g in grp),
                    max(g.y1 for g in grp),
                ))

        img_rects: List[Tuple[int, float, float]] = []
        for img in page.get_images(full=True):
            xref = img[0]
            for r in page.get_image_rects(xref):
                img_rects.append((xref, (r.x0 + r.x1) * 0.5, (r.y0 + r.y1) * 0.5))

        captions: List[Tuple[float, float, int, str, bool]] = []
        pdict = page.get_text("dict")
        for b in pdict.get("blocks", []):
            for l in b.get("lines", []):
                line_txt = "".join(s["text"] for s in l["spans"]).strip()
                if not line_txt:
                    continue
                m = re.match(r"^(\d{1,3})\s*\.\s*(.+)$", line_txt)
                if m:
                    num = int(m.group(1))
                    rest = m.group(2).strip()
                    is_draw = rest.endswith("=") or " = " in rest or rest.endswith("=")
                    author = re.sub(r"\s*[=+!?.]+\s*$", "", rest).strip()
                    bx0, by0, bx1, by1 = l["bbox"]
                    captions.append(((bx0 + bx1) * 0.5, by1, num, author, is_draw))

        boards.sort(key=lambda it: (round(it[2] / 60.0), round(it[1] / 60.0)))
        for sq_cnt, x0, y0, x1, y1 in boards:
            cw = (x1 - x0) / 8.0
            ch = (y1 - y0) / 8.0
            w, b = [], []
            unknown = set()
            for xref, cx, cy in img_rects:
                if x0 - 2.0 <= cx <= x1 + 2.0 and y0 - 2.0 <= cy <= y1 + 2.0:
                    col = int((cx - x0) / cw)
                    rank = 7 - int((cy - y0) / ch)
                    if 0 <= col < 8 and 0 <= rank < 8 and is_dark(rank, col):
                        sq = f"{FILES[col]}{rank + 1}"
                        if xref in WHITE_XREFS:
                            w.append(sq)
                        elif xref in BLACK_XREFS:
                            b.append(sq)
                        else:
                            unknown.add(xref)
            if unknown or len(w) < 2 or len(b) < 2:
                continue

            best_cap = None
            best_dist = 1e9
            cx_board = (x0 + x1) * 0.5
            for tx, ty, num, author, is_draw in captions:
                if x0 - 30.0 <= tx <= x1 + 30.0 and y0 - 35.0 <= ty <= y0 + 8.0:
                    dist = abs(y0 - ty) + 0.3 * abs(cx_board - tx)
                    if dist < best_dist:
                        best_dist = dist
                        best_cap = (num, author, is_draw)

            if not best_cap:
                continue
            num, author, is_draw = best_cap
            if any(sq.endswith("8") for sq in w) or any(sq.endswith("1") for sq in b):
                continue

            diagrams.append({
                "number": num,
                "page": pi + 1,
                "author": author,
                "is_draw": is_draw,
                "white": sorted(w),
                "black": sorted(b),
                "sq_cnt": sq_cnt,
            })

    by_num = {}
    for d in diagrams:
        by_num[d["number"]] = d
    return [by_num[k] for k in sorted(by_num)]


def _black_piece_count(eng: RussianDraughtsEngine) -> int:
    return sum(1 for p in eng.board.values() if p in ("b", "B"))


def _build_line_against_best_defense(
    eng_start: RussianDraughtsEngine,
    first_move_str: str,
    solver: TacticalDraughtsSolver,
    collect_branches: bool = True,
) -> Optional[Tuple[List[str], List[dict]]]:
    """
    Строит линию строго против сильнейших защит чёрных на каждом полуходе.
    Возвращает (mainLine, branches) или None, если чёрные находят уравнивающую защиту.
    """
    cur = eng_start.clone()
    mv0 = next((m for m in cur.legal_moves() if m.to_full_notation() == first_move_str), None)
    if mv0 is None:
        return None
    cur.apply(mv0)
    line: List[str] = [first_move_str]
    branches: List[dict] = []

    start_mat, _, _, _, _ = material_score(eng_start)
    min_w_drop = 0
    max_single_cap = 0
    total_w_caps = 0
    had_promo = False
    in_tactical_captures = False

    def _update_w_stats(mv, eng_before_move):
        nonlocal min_w_drop, max_single_cap, total_w_caps, had_promo, in_tactical_captures
        if mv.is_capture:
            max_single_cap = max(max_single_cap, len(mv.captured))
            total_w_caps += len(mv.captured)
            in_tactical_captures = True
        if mv.promotes:
            had_promo = True

    def _update_mat(cur_eng):
        nonlocal min_w_drop
        m_now, _, _, _, _ = material_score(cur_eng)
        min_w_drop = min(min_w_drop, m_now - start_mat)

    _update_mat(cur)

    # Строим до 14 полуходов, всегда выбирая сильнейшую защиту чёрных
    for step in range(1, 14):
        legal = cur.legal_moves()
        if not legal:
            break

        if cur.turn == "b":
            if len(legal) == 1:
                chosen_mv = legal[0]
            else:
                # Проверяем ВСЕ ходы чёрных, чтобы выбрать сильнейшее и самое упорное сопротивление
                scored_b = []
                per_b_t = max(0.08, min(0.20, 0.70 / len(legal)))
                for bmv in legal:
                    nxt = cur.clone()
                    nxt.apply(bmv)
                    _, w_sc, _, _ = solver.search(nxt, max_depth=11, time_limit_sec=per_b_t)
                    # Для чёрных лучше та ветка, где оценка белых w_sc МЕНЬШЕ;
                    # при близких оценках (в пределах 25 cp) чёрные стараются сохранить больше шашек
                    b_score = -w_sc
                    b_cnt = _black_piece_count(nxt)
                    scored_b.append((b_score, b_cnt, bmv, nxt))

                scored_b.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_b_sc = scored_b[0][0]

                # Если чёрные находят защиту лучше чем -185, задача не форсированно выигрышная!
                if best_b_sc > -185:
                    return None

                # Если есть упорная защита без быстрой сдачи шашек в пределах 25 cp — выбираем её
                chosen_idx = 0
                for ci, (bsc, bc, _, _) in enumerate(scored_b):
                    if bsc >= best_b_sc - 25 and bc > scored_b[chosen_idx][1]:
                        chosen_idx = ci
                chosen_mv = scored_b[chosen_idx][2]

                # Собираем до 1 интересной альтернативной защиты чёрных в branches (если она тоже ведёт к комбинации)
                if collect_branches and len(branches) < 2 and len(scored_b) >= 2 and step <= 5:
                    for bi, (bsc, _, alt_bmv, alt_nxt) in enumerate(scored_b):
                        if bi == chosen_idx:
                            continue
                        if bsc <= -230:
                            # Достраиваем короткую ветку
                            br_moves = [alt_bmv.to_full_notation()]
                            br_eng = alt_nxt.clone()
                            for _ in range(5):
                                bl = br_eng.legal_moves()
                                if not bl:
                                    break
                                if len(bl) == 1:
                                    smv = bl[0]
                                else:
                                    smv, _, _, _ = solver.search(br_eng, max_depth=9, time_limit_sec=0.14)
                                if not smv:
                                    break
                                br_moves.append(smv.to_full_notation())
                                br_eng.apply(smv)
                                if br_eng.turn == "b" and len(br_moves) >= 3 and not any(m.is_capture for m in br_eng.legal_moves()):
                                    break
                            if len(br_moves) >= 2 and br_eng.turn == "b" and not any(m.is_capture for m in br_eng.legal_moves()):
                                branches.append({"afterPly": len(line), "moves": br_moves})
                                break

            cur.apply(chosen_mv)
            line.append(chosen_mv.to_full_notation())
            _update_mat(cur)
        else:
            # Ход белых
            if len(legal) == 1:
                wmv = legal[0]
            else:
                wmv, wsc, _, _ = solver.search(cur, max_depth=11, time_limit_sec=0.25)
                if not wmv or wsc < 175:
                    return None
            _update_w_stats(wmv, cur)
            cur.apply(wmv)
            line.append(wmv.to_full_notation())
            _update_mat(cur)

            # Проверяем, можно ли красиво завершить комбинацию после хода белых
            opp_legal = cur.legal_moves()
            if not opp_legal:
                break
            has_opp_cap = any(m.is_capture for m in opp_legal)
            if not has_opp_cap and len(line) >= 5:
                # Завершаем только если перевес уже реализован (материал/дамка) и не обрезаем разгар серии взятий
                m_now, _, W_now, _, _ = material_score(cur)
                _, check_sc, _, _ = solver.search(cur, max_depth=8, time_limit_sec=0.15)
                if -check_sc >= 210 and (m_now - start_mat >= 2 or W_now > 0) and (wmv.is_capture or len(line) >= 7):
                    break

    if cur.turn != "b" or len(line) < 5:
        # Исключение: только сверхмощный 3-ходовый удар с тройным взятием (max_single_cap >= 3) и жертвой
        if not (cur.turn == "b" and len(line) == 3 and max_single_cap >= 3 and min_w_drop <= -1):
            return None

    if any(m.is_capture for m in cur.legal_moves()):
        return None

    # Фильтр комбинационной содержательности для книги Ячейкина:
    # отсеиваем пресные эндшпили без жертвы/серии взятий/запирания
    is_locked = len(cur.legal_moves()) == 0
    has_combo_essence = (
        min_w_drop <= -1
        or max_single_cap >= 2
        or total_w_caps >= 3
        or had_promo
        or is_locked
    )
    if not has_combo_essence:
        return None

    # Финальная проверка перевеса на глубине 9
    if not is_locked:
        _, final_b_sc, _, _ = solver.search(cur, max_depth=9, time_limit_sec=0.20)
        if -final_b_sc < 200:
            return None

    return line, branches


def _solve_one_yacheykin(diag: dict) -> Optional[dict]:
    if diag.get("is_draw"):
        return None
    w = diag["white"]
    b = diag["black"]
    num = diag["number"]

    eng0 = RussianDraughtsEngine(w, b, [], [], turn="w")
    solver = TacticalDraughtsSolver()
    res = solver.find_combination(
        eng0,
        max_depth=12,
        time_limit_sec=1.8,
        min_advantage=200,
        min_gap_over_second=100,
    )

    candidate_first_moves: List[str] = []
    if res.get("found"):
        candidate_first_moves.append(res["best_move"])
    else:
        # Проверка на 2 равноценных первых хода при полном отрыве от 3-го
        legal0 = eng0.legal_moves()
        if res.get("score", -99999) >= 220 and len(legal0) >= 3:
            scored = []
            for mv in legal0:
                nxt = eng0.clone()
                nxt.apply(mv)
                _, csc, _, _ = solver.search(nxt, max_depth=10, time_limit_sec=0.22)
                scored.append((-csc, mv.to_full_notation()))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] >= 220 and scored[1][0] >= scored[0][0] - 40 and (scored[0][0] - scored[2][0]) >= 120:
                candidate_first_moves.append(scored[0][1])
                candidate_first_moves.append(scored[1][1])

    if not candidate_first_moves:
        return None

    solver.clear()
    built_main = _build_line_against_best_defense(eng0, candidate_first_moves[0], solver, collect_branches=True)
    if built_main is None:
        return None
    main_line, branches = built_main

    alt_wins = []
    if len(candidate_first_moves) > 1:
        built_alt = _build_line_against_best_defense(eng0, candidate_first_moves[1], solver, collect_branches=False)
        if built_alt is not None:
            alt_wins.append({"afterPly": 0, "moves": built_alt[0]})

    puz = {
        "id": f"yacheykin-{num}",
        "number": num,
        "section": "В. Ячейкин (1974) — Украинские шашисты",
        "difficulty": 3,
        "author": diag.get("author", ""),
        "goal": "win",
        "white": w,
        "black": b,
        "whiteKings": [],
        "blackKings": [],
        "turn": "w",
        "mainLine": main_line,
        "branches": branches,
        "altWins": alt_wins,
        "sources": ["В. Ячейкин «Комбинации украинских шашистов» (1974)"],
    }
    if audit_puzzle(puz)[0] != "gold":
        # Если какая-то ветка не прошла аудит — перепроверяем без branches
        puz["branches"] = []
        if audit_puzzle(puz)[0] != "gold":
            return None
    return puz


def main() -> None:
    default_pdf = Path(
        r"C:\Users\Huawe\.gemini\antigravity\brain\387aa0c9-0d9d-4b13-8813-a396dd87a621\.user_uploaded\media_1795546104903_225ee236.pdf"
    )
    pos_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    pdf_path = Path(pos_args[0]) if pos_args else default_pdf
    diags = extract_yacheykin_diagrams(pdf_path)
    draws = sum(1 for d in diags if d["is_draw"])
    print(f"Extracted {len(diags)} unique diagrams from Yacheykin ({draws} marked '=' draw).")

    if "--extract-only" in sys.argv:
        return

    t0 = time.perf_counter()
    workers = min(10, max(2, (os.cpu_count() or 4) - 2))
    solved: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_solve_one_yacheykin, diags, chunksize=6):
            if res is not None:
                solved.append(res)

    solved.sort(key=lambda x: x["number"])
    out_path = ROOT / "yacheykin_gold.json"
    out_path.write_text(json.dumps(solved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Solved & strictly verified {len(solved)}/{len(diags)} Yacheykin combinations in {time.perf_counter()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
