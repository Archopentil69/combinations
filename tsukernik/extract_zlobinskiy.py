#!/usr/bin/env python3
"""
Экстрактор, OCR-парсер и многопоточный решатель для книги
А. С. Злобинского «Комбинации в русских шашках» (1985).
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
from enrich_and_build import fix_rank8_or_rank1_kings, is_complete_clean_combination, solve_tokens_with_lookahead
from extract_sominskiy import decode_ocr_num, detect_tight_boards, normalize_winocr_draughts, refine_inner_8x8_v2
from extract_yacheykin import _build_line_against_best_defense
from tsukernik_validator import RussianDraughtsEngine, is_dark, parse_solution_body, rc_to_sq

SECTION_NAME = "А. Злобинский (1985) — Комбинации"
SOURCE_NAME = "А. С. Злобинский «Комбинации в русских шашках» (1985)"


def extract_zlobinsky_solutions(z_main: dict, z_sol: dict) -> Dict[int, dict]:
    """Собирает все нумерованные решения (и авторов) из конца книги и текстовых страниц."""
    raw_chunks: Dict[int, Tuple[str, List[str]]] = {}
    for ocr_dict in (z_sol, z_main):
        cur_num: Optional[int] = None
        for pname in sorted(ocr_dict.keys()):
            for ln in ocr_dict[pname].get("lines", []):
                txt = ln["text"].strip()
                if not txt or re.fullmatch(r"[0-9ЗзбОІIl]{1,3}", txt):
                    continue
                m = re.match(r"^([0-9ЗзбОІIl]{1,3})\s*[.,]\s+(.+)$", txt)
                if m:
                    n = decode_ocr_num(m.group(1))
                    rest = m.group(2).strip()
                    looks_header = (
                        n is not None
                        and 1 <= n <= 800
                        and ("1." in rest or "1," in rest or "І." in rest or re.match(r"^[А-ЯЁ]\.\s*[А-ЯЁ]", rest))
                        and not (n <= 20 and cur_num is not None and abs(n - cur_num) > 5)
                    )
                    if looks_header:
                        cur_num = n
                        author = ""
                        idx1 = rest.find("1.")
                        if idx1 > 2:
                            cand_a = rest[:idx1].strip(" .,:;-")
                            if re.search(r"[А-ЯЁ]{2}", cand_a) and len(cand_a) <= 35:
                                author = cand_a
                        if n not in raw_chunks:
                            raw_chunks[n] = (author, [rest])
                        continue
                if cur_num is not None and cur_num in raw_chunks:
                    raw_chunks[cur_num][1].append(txt)

    sols: Dict[int, dict] = {}
    for n, (author, parts) in raw_chunks.items():
        joined = " ".join(parts)
        is_draw = bool(re.search(r"\bничь[яюей]\b", joined.lower()))
        norm = normalize_winocr_draughts(joined)
        sols[n] = {"author": author, "raw": joined, "norm": norm, "is_draw": is_draw}
    return sols


def extract_boards_from_zlobinsky(pdf_path: Path, z_main: dict, tmpl: dict) -> List[dict]:
    doc = fitz.open(str(pdf_path))
    boards_list: List[dict] = []
    seen_nums = set()
    auto_num = 1000

    for pi in range(min(228, len(doc))):
        pix = doc[pi].get_pixmap(dpi=200)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = arr[:, :, 0] if pix.n == 1 else cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        boxes = detect_tight_boards(gray)
        if not boxes:
            continue

        pname = f"p_{pi+1:03d}"
        ocr_lines = z_main.get(pname, {}).get("lines", [])
        scale = 200.0 / 135.0
        text_items = []
        for ln in ocr_lines:
            x0, y0, x1, y1 = [c * scale for c in ln["bbox"]]
            text_items.append(((x0 + x1) * 0.5, (y0 + y1) * 0.5, ln["text"].strip()))

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
                    core = patch[16:36, 12:36].astype(np.float32)
                    c_p30 = float(np.percentile(core, 30))
                    c_p85 = float(np.percentile(core, 85))

                    lbl = "."
                    best_s = scores["empty"]
                    if scores["W"] > best_s and scores["W"] >= 0.48 and scores["W"] > scores["w"] + 0.04 and c_p85 >= 188:
                        best_s = scores["W"]
                        lbl = "W"
                    elif scores["w"] > best_s and scores["w"] >= 0.42 and c_p85 >= 188:
                        best_s = scores["w"]
                        lbl = "w"
                    if scores["b"] > best_s and scores["b"] >= 0.44 and c_p30 < 80:
                        lbl = "b"

                    sq = rc_to_sq(r, col)
                    if lbl == "w":
                        w.append(sq)
                    elif lbl == "b":
                        b.append(sq)
                    elif lbl == "W":
                        wk.append(sq)

            if len(w) + len(wk) < 2 or len(b) + len(bk) < 2:
                continue

            # Ищем номер прямо под доской (как 317..322 на p100) или над ней
            cx_b = bx + bw * 0.5
            num: Optional[int] = None
            best_d = 1e9
            for tx, ty, txt in text_items:
                if bx - 25 <= tx <= bx + bw + 25 and by + bh - 5 <= ty <= by + bh + 70:
                    m_num = re.fullmatch(r"[^0-9ЗзбОІIl]*([0-9ЗзбОІIl]{1,3})[^0-9ЗзбОІIl]*", txt)
                    if m_num:
                        val = decode_ocr_num(m_num.group(1))
                        if val is not None and 1 <= val <= 800:
                            dist = abs(ty - (by + bh + 20)) + 0.4 * abs(cx_b - tx)
                            if dist < best_d:
                                best_d = dist
                                num = val
            if num is None or num in seen_nums:
                auto_num += 1
                num = auto_num
            else:
                seen_nums.add(num)

            boards_list.append({
                "number": num,
                "page": pi + 1,
                "white": sorted(w),
                "black": sorted(b),
                "whiteKings": sorted(wk),
                "blackKings": sorted(bk),
            })
    return boards_list


def _solve_one_zlobinskiy(args_tuple) -> Optional[dict]:
    item, sol_info = args_tuple
    num = item["number"]
    w = list(item["white"])
    b = list(item["black"])
    wk = list(item["whiteKings"])
    bk = list(item["blackKings"])
    author = (sol_info or {}).get("author", "")

    solver = TacticalDraughtsSolver()

    # 1. Сверка с OCR-решением из конца книги Злобинского
    if sol_info and not sol_info.get("is_draw"):
        norm = sol_info["norm"]
        idx1 = norm.find("1.")
        body = norm[idx1:] if idx1 >= 0 else norm
        parsed = parse_solution_body(num, body)
        if parsed and parsed.candidates:
            good_cands = [c for c in parsed.candidates if len(c.main_tokens) >= 4] or parsed.candidates[:1]
            w_clean = [s for s in w if not s.endswith("8")]
            wk_auto = list(set(wk) | {s for s in w if s.endswith("8")})
            b_clean = [s for s in b if not s.endswith("1")]
            bk_auto = list(set(bk) | {s for s in b if s.endswith("1")})

            configs = [(w_clean, b_clean, wk_auto, bk_auto)]
            for sq in w_clean:
                configs.append(([x for x in w_clean if x != sq], b_clean, wk_auto + [sq], bk_auto))
            for sq in b_clean:
                configs.append((w_clean, [x for x in b_clean if x != sq], wk_auto, bk_auto + [sq]))
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
                            "id": f"zlobinskiy-{num}",
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

    # 2. Фолбэк: авторешение движком (только если не помечена ничья и есть чистая выигрывающая комбинация против лучшей защиты!)
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
        "id": f"zlobinskiy-{num}",
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
    pdf_path = ROOT / "Zlobinskiy_Kombinatsii_v_russkih_shashkah_1985.pdf"
    z_main = json.loads((SCR / "zlob_main_ocr.json").read_text("utf-8-sig"))
    z_sol = json.loads((SCR / "zlob_sol_ocr.json").read_text("utf-8-sig"))
    npz = np.load(str(SCR / "zlob_tmpl.npz"))
    tmpl = {k: npz[k] for k in ("w", "b", "W", "empty")}

    sols = extract_zlobinsky_solutions(z_main, z_sol)
    boards = extract_boards_from_zlobinsky(pdf_path, z_main, tmpl)
    print(f"Zlobinsky extracted: {len(boards)} diagrams, {len(sols)} OCR solutions.")

    t0 = time.perf_counter()
    workers = min(10, max(2, (os.cpu_count() or 4) - 2))
    tasks = [(b, sols.get(b["number"])) for b in boards]
    solved: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_solve_one_zlobinskiy, tasks, chunksize=8):
            if r is not None:
                solved.append(r)

    solved.sort(key=lambda x: x["number"])
    books_dir = ROOT / "books"
    books_dir.mkdir(exist_ok=True)
    out_path = books_dir / "zlobinskiy.json"
    out_path.write_text(json.dumps(solved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Solved & strictly verified {len(solved)}/{len(boards)} Zlobinsky puzzles in {time.perf_counter()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
