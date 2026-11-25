#!/usr/bin/env python3
"""
Автоматическая поглавная оцифровка всей книги Э. Г. Цукерника
(стр. 12–368) с сохранением прогресса и итоговой сборкой без дубликатов.

Запуск:
  python run_all_book.py
"""

import subprocess
import sys
from pathlib import Path

PDF_FILE = "drafts_combinations.pdf"
CHAPTERS_DIR = Path("chapters")
REVIEWS_DIR = Path("reviews")
DEBUG_DIR = Path("debug_crops")

# Все главы с комбинациями по оглавлению книги (стр. 12–368)
CHAPTERS = [
    ("01_a1_a7", 12, 29),
    ("02_c1_c3", 29, 42),
    ("03_e1_e5", 42, 60),
    ("04_g1_g5", 60, 75),
    ("05_b2", 75, 97),
    ("06_b4_b6", 97, 117),
    ("07_d2", 117, 135),
    ("08_d4_d6", 135, 147),
    ("09_f2", 147, 175),
    ("10_f4_f6", 175, 185),
    ("11_h2", 185, 228),
    ("12_h4", 228, 286),
    ("13_h6", 286, 315),
    ("14_raznye_udary", 315, 345),
    ("15_zapiranie_cugcvang_samoobl", 345, 354),
    ("16_kontrpriemy", 354, 369),
]


def main() -> None:
    if not Path(PDF_FILE).exists():
        print(f"Ошибка: файл {PDF_FILE} не найден в текущей папке.")
        sys.exit(1)

    CHAPTERS_DIR.mkdir(exist_ok=True)
    REVIEWS_DIR.mkdir(exist_ok=True)

    json_files = []

    for name, start_p, end_p in CHAPTERS:
        out_json = CHAPTERS_DIR / f"ch_{name}.json"
        out_rev = REVIEWS_DIR / f"rev_{name}.json"
        json_files.append(str(out_json))

        if out_json.exists() and out_json.stat().st_size > 10:
            print(f"[SKIP] Глава {name} (стр. {start_p}-{end_p}) уже готова -> {out_json}")
            continue

        print(f"\n>>> Обработка главы {name} (стр. {start_p}-{end_p})...")
        cmd = [
            sys.executable,
            "tsukernik_validator.py",
            "pdf",
            PDF_FILE,
            "--pages",
            f"{start_p}-{end_p}",
            "--out",
            str(out_json),
            "--review",
            str(out_rev),
            "--debug-dir",
            str(DEBUG_DIR / name),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[WARN] Глава {name} завершилась с ошибкой (код {res.returncode}), идём дальше.")

    # Склеиваем все готовые главы через pdn_builder.py (дедупликация + теги + рейтинг)
    existing_jsons = [f for f in json_files if Path(f).exists() and Path(f).stat().st_size > 10]
    if existing_jsons and Path("pdn_builder.py").exists():
        print("\n>>> Сборка общей мастер-базы без дубликатов...")
        cmd_merge = [
            sys.executable,
            "pdn_builder.py",
            "build",
            *existing_jsons,
            "--out",
            "tsukernik_full_master.json",
        ]
        subprocess.run(cmd_merge)
    else:
        print("\nГотово! Файлы по главам сохранены в папке chapters/.")


if __name__ == "__main__":
    main()