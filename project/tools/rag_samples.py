"""Deterministic real file fixtures. Run in the existing document-parser env.

These are small authored test documents, not fabricated OCR output or human gold.
"""
import datetime
import json
import sys
from pathlib import Path


def build(directory, amount="128.50"):
    from PIL import Image, ImageDraw, ImageFont
    from docx import Document
    from docx.shared import Inches
    from openpyxl import Workbook

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1000, 460), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    for i, text in enumerate(["RAG Receipt Verification", "Receipt: IMG-001", f"Total amount: {amount} CNY", "Status: PAID", "Date: 2026-11-24"]):
        draw.text((40, 30 + i * 78), text, fill="black", font=font)
    image.save(out / "receipt.png")
    image.save(out / "receipt.jpg", quality=95)
    doc = Document()
    doc.add_heading("Equipment Handbook", 0)
    doc.add_heading("Warranty", 1)
    doc.add_paragraph("The warranty period is 24 months. Refunds require the original receipt.")
    table = doc.add_table(rows=1, cols=3)
    for c, text in zip(table.rows[0].cells, ["Metric", "Model A", "Model B"]):
        c.text = text
    for values in [["Recall", "85%", "92%"], ["Cost", "10 CNY", "15 CNY"]]:
        for c, text in zip(table.add_row().cells, values):
            c.text = text
    doc.add_paragraph("After the comparison table.")
    doc.add_picture(str(out / "receipt.png"), width=Inches(5))
    doc.add_paragraph("Figure 1: purchase receipt.", style="Caption")
    doc.add_heading("Merged heading example", 1)
    merged = doc.add_table(rows=3, cols=3)
    merged.cell(0, 0).text = "Object"
    merged.cell(0, 1).merge(merged.cell(0, 2)).text = "Metrics"
    for c, text in zip(merged.rows[1].cells, ["Object", "Accuracy", "Recall"]):
        c.text = text
    for c, text in zip(merged.rows[2].cells, ["Model C", "91%", "87%"]):
        c.text = text
    doc.save(out / "handbook.docx")
    wb = Workbook()
    models = wb.active
    models.title = "Models"
    models.append(["Model", "Accuracy", "Recall"])
    models.append(["LogReg", 0.90, 0.80])
    models.append(["SVM", 0.95, 0.85])
    for row in models.iter_rows(min_row=2, min_col=2):
        for c in row:
            c.number_format = "0.00%"
    budget = wb.create_sheet("Budget")
    budget.append(["Item", "Amount", "Date"])
    budget.append(["Laptop", 1200.50, datetime.date(2026, 11, 24)])
    budget.append(["Cable", 20, datetime.date(2026, 11, 25)])
    budget["B2"].number_format = '"¥"0.00'
    budget["B3"].number_format = '"¥"0.00'
    budget["F1"] = "Formula (not computed)"
    budget["F2"] = "=B2+B3"
    merged_sheet = wb.create_sheet("Merged")
    merged_sheet.append(["Object", "Metrics", None])
    merged_sheet.merge_cells("B1:C1")
    merged_sheet.append(["Object", "Accuracy", "Recall"])
    merged_sheet.append(["Model C", 0.91, 0.87])
    for c in [merged_sheet["B3"], merged_sheet["C3"]]:
        c.number_format = "0.00%"
    wb.save(out / "measurements.xlsx")
    (out / "fixture_manifest.json").write_text(json.dumps({"kind": "deterministic authored test fixtures, AI annotations pending human", "receipt_amount": amount, "receipt_status": "PAID", "warranty_months": 24, "models": {"LogReg": {"accuracy": "90.00%", "recall": "80.00%"}, "SVM": {"accuracy": "95.00%", "recall": "85.00%"}}, "formula": "uncached; must not evaluate"}, indent=2))


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "128.50")
