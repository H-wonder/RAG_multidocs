"""Author clearly labeled OCR verification files; no reference OCR is injected."""

import io
import json
from pathlib import Path
import sys


def build(directory):
    import pymupdf as fitz
    from PIL import Image, ImageDraw, ImageFont

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 32)

    def image(lines):
        picture = Image.new("RGB", (1100, 800), "white")
        draw = ImageDraw.Draw(picture)
        for i, line in enumerate(lines):
            draw.text((50, 35 + i * 78), line, fill="black", font=font)
        return picture

    first = image(
        [
            "校园采购单 Campus Purchase",
            "编号 Receipt: SCAN-A1",
            "日期 Date: 2026-11-25",
            "金额 Amount: 246.80 CNY",
            "准确率 Accuracy: 93.5%",
            "项目 Item     数量 Qty     金额 Amount",
            "螺丝 Screw    5            20.50 CNY",
            "状态 Status: PAID",
        ]
    )
    second = image(
        [
            "设备使用说明 Equipment Notice",
            "开放时间: 每天24小时",
            "Warranty period: 36 months",
            "编号 Receipt: SCAN-B2",
            "日期 Date: 2026-11-26",
            "费用 Fee: 0 CNY",
        ]
    )
    first.save(out / "bilingual.png")
    first.save(out / "bilingual.jpg", quality=95)

    def scan(page, picture, rect=None):
        data = io.BytesIO()
        picture.save(data, format="PNG")
        page.insert_image(rect or page.rect, stream=data.getvalue())

    def native(doc):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 65), "Native Verification - NATIVE-01", fontsize=18)
        page.insert_text((50, 105), "Warranty: 48 months. Native amount: 75.25 CNY.", fontsize=12)
        return page

    for name in ["native", "scanned", "multi_scanned", "mixed_pages", "hybrid", "bad_layer"]:
        doc = fitz.open()
        if name == "native":
            native(doc)
        elif name == "scanned":
            scan(doc.new_page(width=595, height=842), first)
        elif name == "multi_scanned":
            scan(doc.new_page(width=595, height=842), first)
            scan(doc.new_page(width=595, height=842), second)
        elif name == "mixed_pages":
            native(doc)
            scan(doc.new_page(width=595, height=842), first)
            native(doc)
        elif name == "hybrid":
            page = native(doc)
            page.insert_text(
                (50, 130),
                "Native text must remain intact. The image below contains separate purchase evidence.",
                fontsize=10,
            )
            scan(page, first, fitz.Rect(35, 180, 560, 740))
        else:
            page = doc.new_page(width=595, height=842)
            scan(page, first)
            page.insert_font(fontname="badfont", fontfile=font_path)
            page.insert_text((30, 30), "\ufffd" * 40, fontname="badfont", fontsize=8, render_mode=3)
        doc.save(out / (name + ".pdf"))
        for i, page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(
                out / f"{name}_page{i + 1}.png"
            )
        doc.close()
    (out / "authored_fields.json").write_text(
        json.dumps(
            {
                "kind": "authored development fixtures; original images require visual verification; not human annotated heldout",
                "purchase": {
                    "id": "SCAN-A1",
                    "date": "2026-11-25",
                    "amount": "246.80 CNY",
                    "accuracy": "93.5%",
                    "status": "PAID",
                },
                "notice": {"hours": "每天24小时", "warranty": "36 months"},
                "native": {"warranty": "48 months", "amount": "75.25 CNY"},
                "bad_layer": "Invisible U+FFFD text layer deliberately embedded; no text values are substituted into OCR results.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    build(sys.argv[1])
