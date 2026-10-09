"""
inspect_vrod.py - prints the structure of the Berkeley Voluntary Registry Offsets
Database (VROD) Excel file, so an ingest adapter can be written against the real
schema instead of guesses.

Default mode reads only sheet names, sizes and the first rows (fast).
--detail mode additionally prints:
  - every non-empty header cell of the PROJECTS sheet (rows 1-4), with column letters
  - the full text of the documentation sheets (READ FIRST, FAQ, Column Descriptions)

Usage:
    pip install openpyxl
    python inspect_vrod.py "data\\VROD.xlsx" > data\\vrod_structure.txt
    python inspect_vrod.py "data\\VROD.xlsx" --detail > data\\vrod_detail.txt
"""
import sys

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

MAX_COLS = 30   # columns shown per row in overview mode
MAX_ROWS = 6    # first rows shown per sheet in overview mode
CELL = 32       # max characters per cell in overview mode
DOC_SHEETS = ["READ FIRST", "FAQ", "Column Descriptions"]
HEADER_SHEET, HEADER_ROWS, HEADER_CELL = "PROJECTS", 4, 80
DOC_CELL = 400


def short(v, n=CELL):
    s = "" if v is None else str(v).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "~"


def overview(wb):
    for name in wb.sheetnames:
        ws = wb[name]
        print("=" * 70)
        print(f"SHEET: {name}   rows ~{ws.max_row}, cols ~{ws.max_column}")
        print("=" * 70)
        for i, row in enumerate(ws.iter_rows(max_row=MAX_ROWS, values_only=True), 1):
            cells = [short(v) for v in row[:MAX_COLS]]
            while cells and cells[-1] == "":
                cells.pop()
            print(f"  row {i}: " + " | ".join(cells))
        print()


def detail(wb):
    if HEADER_SHEET in wb.sheetnames:
        print("=" * 70)
        print(f"ALL NON-EMPTY HEADER CELLS: {HEADER_SHEET}, rows 1-{HEADER_ROWS}")
        print("=" * 70)
        ws = wb[HEADER_SHEET]
        for r, row in enumerate(ws.iter_rows(max_row=HEADER_ROWS, values_only=True), 1):
            for c, v in enumerate(row, 1):
                if v not in (None, ""):
                    print(f"  r{r} {get_column_letter(c)}: {short(v, HEADER_CELL)}")
        print()
    for name in DOC_SHEETS:
        if name not in wb.sheetnames:
            continue
        print("=" * 70)
        print(f"FULL TEXT: {name}")
        print("=" * 70)
        for i, row in enumerate(wb[name].iter_rows(values_only=True), 1):
            cells = [short(v, DOC_CELL) for v in row if v not in (None, "")]
            if cells:
                print(f"  {i}: " + " | ".join(cells))
        print()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return
    wb = load_workbook(args[0], read_only=True, data_only=True)
    print(f"File: {args[0]}")
    print(f"Sheets: {len(wb.sheetnames)}\n")
    if "--detail" in sys.argv:
        detail(wb)
    else:
        overview(wb)
    wb.close()


if __name__ == "__main__":
    main()
