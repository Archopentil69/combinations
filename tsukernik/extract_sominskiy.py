#!/usr/bin/env python3
"""
Экстрактор, OCR-парсер и многопоточный решатель для книги
В. Г. Соминского «Комбинационные идеи в русских шашках».
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

import cv2
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parent
SCR = Path(r"C:\Users\Huawe\.gemini\antigravity\brain\387aa0c9-0d9d-4b13-8813-a396dd87a621\scratch")
sys.path.insert(0, str(ROOT))

from audit_master import audit_puzzle
from draughts_solver import TacticalDraughtsSolver
from enrich_and_build import clean_raw_solution_text, fix_rank8_or_rank1_kings, is_complete_clean_combination, solve_tokens_with_lookahead
from extract_yacheykin import _build_line_against_best_defense
from tsukernik_validator import FILES, RussianDraughtsEngine, is_dark, parse_solution_body, rc_to_sq, sort_boxes_by_rows

SECTION_NAME = "В. Соминский — Комбинационные идеи"
SOURCE_NAME = "В. Г. Соминский «Комбинационные идеи в русских шашках»"


def detect_tight_boards(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    bw = (gray < 175).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hp, wp = gray.shape
    raw = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 0.88 <= (w / float(h or 1)) <= 1.12 and (wp * 0.16) <= w <= (wp * 0.42):
            raw.append((x, y, w, h))
    if len(raw) < 2:
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k2)
        for c in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(c)
            if 0.88 <= (w / float(h or 1)) <= 1.12 and (wp * 0.16) <= w <= (wp * 0.42):
                if not any(abs(x - rx) < 25 and abs(y - ry) < 25 for rx, ry, _, _ in raw):
                    raw.append((x, y, w, h))
    return sort_boxes_by_rows(raw)


def refine_inner_8x8_v2(board_gray: np.ndarray) -> np.ndarray:
    h, w = board_gray.shape
    dark = (board_gray < 165).astype(np.float32)
    row_m, col_m = dark.mean(axis=1), dark.mean(axis=0)
    max_t = max(4, int(min(h, w) * 0.055))

    def find_edge(prof, indices):
        last_line = None
        for idx in indices:
            if prof[idx] > 0.58:
                last_line = idx
            elif last_line is not None and prof[idx] < 0.35:
                return idx
        return (last_line + 1) if last_line is not None else max(2, int(len(prof) * 0.025))

    top = find_edge(row_m, range(0, max_t))
    bottom = h - 1 - find_edge(row_m[::-1], range(0, max_t))
    left = find_edge(col_m, range(0, max_t))
    right = w - 1 - find_edge(col_m[::-1], range(0, max_t))
    return board_gray[top:bottom + 1, left:right + 1]


OCR_DIGIT_TRANS = str.maketrans({
    "З": "3", "з": "3", "б": "6", "О": "0", "о": "0", "І": "1", "і": "1", "l": "1", "I": "1"
})
OCR_MOVE_TRANS = str.maketrans({
    "а": "a", "А": "a",
    "Ь": "b", "ь": "b", "Б": "b",
    "с": "c", "С": "c",
    "е": "e", "Е": "e",
    "д": "g", "Д": "g",
    "З": "3", "з": "3",
    "б": "6",
    "І": "1", "і": "1",
    "О": "0",
})


def decode_ocr_num(tok: str) -> Optional[int]:
    t = tok.strip().translate(OCR_DIGIT_TRANS)
    return int(t) if t.isdigit() else None


def normalize_winocr_draughts(s: str) -> str:
    """Нормализует типичные подмены кириллицы/латиницы и пробелы от Windows OCR в шашечной нотации."""
    s = s.replace("—", "-").replace("–", "-").replace("×", "x")
    s = re.sub(r"(\d|[ЗІ])\s*\.\s*\.\s*\.?", lambda m: m.group(1).translate(OCR_DIGIT_TRANS) + "... ", s)
    s = re.sub(r"\s*[:;•·]\s*", ":", s)
    s = re.sub(r"\s*-\s*", "-", s)
    # Удаляем длинные словесные комментарии в скобках, если в них нет ходов
    s = re.sub(r"\(([^()]{12,})\)", lambda m: "" if not re.search(r"[:\-]\s*[a-hаесЬд][1-8Зб]", m.group(1), re.I) else m.group(0), s)

    tokens = s.split()
    out = []
    for tok in tokens:
        t_low = tok.translate(OCR_MOVE_TRANS)
        if re.search(r"[жзийклмнопрстуфхцчшщъыэюя]{2}", tok.lower()) and ":" not in tok and "-" not in tok:
            out.append(tok)
            continue
        if re.match(r"^\d{1,2}\.$", t_low):
            out.append(t_low)
            continue
        m_trail = re.match(
            r"^([a-h16]{2,3}(?:[:\-][a-h16][1-8a-h]?)+|[a-h16]{2}[1-8Зб])[1Itl!хxХ]*([.,;)]*)$",
            t_low,
            re.I,
        )
        if m_trail:
            out.append(m_trail.group(1).lower() + m_trail.group(2))
            continue
        if re.search(r"[:\-]", t_low) and re.search(r"[a-h]", t_low, re.I) and len(t_low) <= 18:
            out.append(t_low.lower())
        else:
            out.append(tok)
    return clean_raw_solution_text(" ".join(out))


def extract_sominsky_solutions(ocr_data: dict) -> Dict[int, dict]:
    """Извлекает нумерованные решения (1..340) из OCR всего тома Соминского."""
    raw_chunks: Dict[int, List[str]] = {}
    cur_num: Optional[int] = None
    last_num = 0

    for pname in sorted(ocr_data.keys()):
        lines = ocr_data[pname].get("lines", [])
        for ln in lines:
            txt = ln["text"].strip()
            if not txt or re.fullmatch(r"[0-9ЗзбОІIl]{1,3}", txt):
                continue
            m = re.match(r"^([0-9ЗзбОІIl]{1,3})\s*[.,]\s+(.+)$", txt)
            if m:
                n = decode_ocr_num(m.group(1))
                rest = m.group(2).strip()
                is_move_num = (
                    n is not None
                    and n <= 15
                    and cur_num is not None
                    and abs(n - cur_num) > 5
                    and bool(re.match(r"^[a-hаесЬд][1-8Зб]", rest))
                )
                looks_like_solution_start = (
                    n is not None
                    and not is_move_num
                    and 1 <= n <= 360
                    and (n > last_num - 4 or n >= 90)
                    and (
                        "1." in rest
                        or "1," in rest
                        or "І." in rest
                        or "В этой" in rest
                        or "Эта позиция" in rest
                        or "У белых" in rest
                        or "У черных" in rest
                        or "Белые" in rest
                    )
                )
                if looks_like_solution_start:
                    cur_num = n
                    last_num = max(last_num, n)
                    raw_chunks.setdefault(n, []).append(rest)
                    continue
            if cur_num is not None:
                if re.match(r"^[0-9ЗзбОІIl]{1,3}\s+[А-ЯЁ]\.\s*[А-ЯЁ]", txt):
                    cur_num = None
                    continue
                raw_chunks[cur_num].append(txt)

    sols: Dict[int, dict] = {}
    for n, parts in raw_chunks.items():
        joined = " ".join(parts)
        is_draw = bool(re.search(r"\bничь[яюей]\b", joined.lower()))
        norm = normalize_winocr_draughts(joined)
        sols[n] = {"raw": joined, "norm": norm, "is_draw": is_draw}
    return sols


def extract_boards_from_sominsky(pdf_path: Path, ocr_data: dict, tmpl: dict) -> List[dict]:
    doc = fitz.open(str(pdf_path))
    boards_list: List[dict] = []
    seen_nums = set()

    for pi in range(min(62, len(doc))):
        pix = doc[pi].get_pixmap(dpi=200)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = arr[:, :, 0] if pix.n == 1 else cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        boxes = detect_tight_boards(gray)
        if not boxes:
            continue

        pname = f"p_{pi+1:03d}"
        ocr_lines = ocr_data.get(pname, {}).get("lines", [])
        # Коэффициент перехода из 160 DPI (OCR) -> 200 DPI (CV)
        scale = 200.0 / 160.0

        # Парсим заголовки: строка с номером '31' или '31 А. Аюпов' и строка автора над доской
        text_items = []
        for ln in ocr_lines:
            x0, y0, x1, y1 = [c * scale for c in ln["bbox"]]
            text_items.append(((x0 + x1) * 0.5, y1, ln["text"].strip()))

        for bx, by, bw, bh in boxes:
            inner = refine_inner_8x8_v2(gray[by:by + bh, bx:bx + bw])
            ih, iw = inner.shape
            cw, ch = iw / 8.0, ih / 8.0

            w, b, wk, bk = [], [], [], []
            for ri in range(8):
                r = 7 - ri
                for col in range(8):
                    if not is_dark(r, col):
                        continue
                    patch = cv2.resize(inner[int(ri * ch):int((ri + 1) * ch), int(col * cw):int((col + 1) * cw)], (48, 48))
                    scores = {lbl: float(cv2.matchTemplate(patch, t[4:44, 4:44], cv2.TM_CCOEFF_NORMED).max()) for lbl, t in tmpl.items()}
                    # Абсолютная проверка яркости, чтобы тёмные штрихи не давали ложных чёрных шашек
                    core = patch[16:34, 14:34].astype(np.float32)
                    c_p50 = float(np.percentile(core, 50))
                    c_p85 = float(np.percentile(core, 85))
                    yy, xx = np.ogrid[:48, :48]
                    dist = np.sqrt((yy - 23.5)**2 + (xx - 23.5)**2)
                    halo_m = float(patch[(dist > 12) & (dist <= 18)].mean())
                    core_m = float(patch[dist <= 10].mean())
                    dark_contrast = halo_m - core_m

                    lbl = "."
                    best_s = scores["empty"]
                    if scores["w"] > best_s and scores["w"] >= 0.44 and c_p85 >= 190:
                        best_s = scores["w"]
                        lbl = "w"
                    if scores["B"] > best_s and scores["B"] >= 0.46 and halo_m < 80 and c_p85 >= 170:
                        best_s = scores["B"]
                        lbl = "B"
                    if scores["b"] > best_s and scores["b"] >= 0.48 and c_p50 < 65 and dark_contrast > 28:
                        lbl = "b"

                    sq = rc_to_sq(r, col)
                    if lbl == "w":
                        w.append(sq)
                    elif lbl == "b":
                        b.append(sq)
                    elif lbl == "B":
                        bk.append(sq)

            if len(w) + len(wk) < 2 or len(b) + len(bk) < 2:
                continue

            # Ищем номер и автора прямо над доской (в окне y = by - 120 .. by + 10)
            cx_b = bx + bw * 0.5
            cand_above = [
                (abs(by - ty) + 0.25 * abs(cx_b - tx), ty, txt)
                for tx, ty, txt in text_items
                if bx - 40 <= tx <= bx + bw + 40 and by - 125 <= ty <= by + 12
            ]
            cand_above.sort(key=lambda x: x[1])
            num: Optional[int] = None
            author = ""
            for _, _, txt in cand_above:
                m_num = re.search(r"(?:^|\s)([0-9ЗзбОІIl]{1,3})(?=\s|$|\.)", txt)
                if m_num and num is None:
                    val = decode_ocr_num(m_num.group(1))
                    if val is not None and 1 <= val <= 360:
                        num = val
                        after = txt[m_num.end():].strip(" .,:;-")
                        if after and re.search(r"[А-ЯЁ]", after):
                            author = after
                        continue
                if re.search(r"[А-ЯЁ]\.\s*[А-ЯЁ]", txt) and not author:
                    author = txt

            if num is None or num in seen_nums:
                continue
            seen_nums.add(num)
            boards_list.append({
                "number": num,
                "page": pi + 1,
                "author": author,
                "white": sorted(w),
                "black": sorted(b),
                "whiteKings": sorted(wk),
                "blackKings": sorted(bk),
            })
    return boards_list


def _solve_one_sominskiy(args_tuple) -> Optional[dict]:
    item, sol_info = args_tuple
    num = item["number"]
    w = list(item["white"])
    b = list(item["black"])
    wk = list(item["whiteKings"])
    bk = list(item["blackKings"])
    author = item.get("author", "")

    solver = TacticalDraughtsSolver()

    # 1. Пытаемся сопоставить с OCR-решением из книги Соминского
    if sol_info and not sol_info.get("is_draw"):
        norm = sol_info["norm"]
        # Вырезаем часть от '1.'
        idx1 = norm.find("1.")
        body = norm[idx1:] if idx1 >= 0 else norm
        parsed = parse_solution_body(num, body)
        if parsed and parsed.candidates:
            good_cands = [c for c in parsed.candidates if len(c.main_tokens) >= 4]
            if not good_cands:
                good_cands = parsed.candidates[:1]

            # Пробуем исходную позицию + перевод любой одиночной фигуры в дамку (W/B) + 1-cell ремонт
            configs = [(w, b, wk, bk)]
            # Если есть белые/чёрные на 8-й/1-й — сразу пробуем дамки
            w_clean = [s for s in w if not s.endswith("8")]
            wk_auto = list(set(wk) | {s for s in w if s.endswith("8")})
            b_clean = [s for s in b if not s.endswith("1")]
            bk_auto = list(set(bk) | {s for s in b if s.endswith("1")})
            if w_clean != w or b_clean != b:
                configs.insert(0, (w_clean, b_clean, wk_auto, bk_auto))
            # Перебор 1 дамки среди имеющихся шашек (часто спасает если W/B прочитана как простая)
            for sq in w_clean:
                configs.append(([x for x in w_clean if x != sq], b_clean, wk_auto + [sq], bk_auto))
            for sq in b_clean:
                configs.append((w_clean, [x for x in b_clean if x != sq], wk_auto, bk_auto + [sq]))
            # Удаление 1 ложной или смена цвета 1 фигуры
            for i in range(len(w_clean)):
                configs.append(([x for k, x in enumerate(w_clean) if k != i], b_clean, wk_auto, bk_auto))
            for i in range(len(b_clean)):
                configs.append((w_clean, [x for k, x in enumerate(b_clean) if k != i], wk_auto, bk_auto))

            for cand in good_cands[:2]:
                for w2, b2, wk2, bk2 in configs:
                    eng = RussianDraughtsEngine(w2, b2, wk2, bk2, turn="w")
                    line = solve_tokens_with_lookahead(eng, cand.main_tokens)
                    if line and len(line) >= 5:
                        puz = {
                            "id": f"sominskiy-{num}",
                            "number": num,
                            "section": SECTION_NAME,
                            "difficulty": 3,
                            "author": author,
                            "goal": "win",
                            "white": sorted(w2),
                            "black": sorted(b2),
                            "whiteKings": sorted(wk2),
                            "blackKings": sorted(bk2),
                            "turn": "w",
                            "mainLine": line,
                            "branches": [],
                            "sources": [SOURCE_NAME],
                        }
                        fixed = fix_rank8_or_rank1_kings(puz)
                        if fixed and audit_puzzle(fixed)[0] == "gold":
                            if is_complete_clean_combination(fixed, solver)[0]:
                                return fixed

    # 2. Фолбэк: если OCR решения не хватило и позиция не ничейная — решаем сильным движком
    if sol_info and sol_info.get("is_draw"):
        return None
    w0 = [s for s in w if not s.endswith("8")]
    b0 = [s for s in b if not s.endswith("1")]
    if len(w0) + len(wk) < 2 or len(b0) + len(bk) < 2:
        return None

    eng0 = RussianDraughtsEngine(w0, b0, wk, bk, turn="w")
    res = solver.find_combination(eng0, max_depth=12, time_limit_sec=1.6, min_advantage=200, min_gap_over_second=100)
    if not res.get("found"):
        return None
    built = _build_line_against_best_defense(eng0, res["best_move"], solver, collect_branches=True)
    if built is None:
        return None
    main_line, branches = built
    puz = {
        "id": f"sominskiy-{num}",
        "number": num,
        "section": SECTION_NAME,
        "difficulty": 3,
        "author": author,
        "goal": "win",
        "white": sorted(w0),
        "black": sorted(b0),
        "whiteKings": sorted(wk),
        "blackKings": sorted(bk),
        "turn": "w",
        "mainLine": main_line,
        "branches": branches,
        "sources": [SOURCE_NAME],
    }
    if audit_puzzle(puz)[0] != "gold":
        puz["branches"] = []
        if audit_puzzle(puz)[0] != "gold":
            return None
    return puz


def main() -> None:
    pdf_path = ROOT / "Sominskiy_kombinatsionnye_idei_v_russkih_shashkah.pdf"
    ocr_data = json.loads((SCR / "som_ocr.json").read_text("utf-8-sig"))
    npz = np.load(str(SCR / "som_tmpl.npz"))
    tmpl = {k: npz[k] for k in ("w", "b", "B", "empty")}

    sols = extract_sominsky_solutions(ocr_data)
    boards = extract_boards_from_sominsky(pdf_path, ocr_data, tmpl)
    print(f"Sominsky extracted: {len(boards)} unique diagrams, {len(sols)} OCR solutions.")

    t0 = time.perf_counter()
    workers = min(10, max(2, (os.cpu_count() or 4) - 2))
    tasks = [(b, sols.get(b["number"])) for b in boards]
    solved: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_solve_one_sominskiy, tasks, chunksize=6):
            if r is not None:
                solved.append(r)

    solved.sort(key=lambda x: x["number"])
    books_dir = ROOT / "books"
    books_dir.mkdir(exist_ok=True)
    out_path = books_dir / "sominskiy.json"
    out_path.write_text(json.dumps(solved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Solved & strictly verified {len(solved)}/{len(boards)} Sominsky puzzles in {time.perf_counter()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
