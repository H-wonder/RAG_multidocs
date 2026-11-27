"""Real local OCR→public Core acceptance with retained source IR and prompts."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid
from dataclasses import replace

from rag_local import PROJECT, PARSER, core, settings
from rag_multidocs.evaluation.metrics import ablation_policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--backend", choices=["keyword", "milvus"], default="milvus")
    args = parser.parse_args()
    out = args.output.resolve()
    samples = args.samples.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    policy = replace(
        ablation_policies()["full"],
        model_context_tokens=4096,
        output_tokens=700,
        request_seconds=120,
    )
    service = core(settings(out, "p1_ocr_" + uuid.uuid4().hex[:10], args.backend), policy)
    result = {
        "backend": args.backend,
        "checks": {},
        "documents": {},
        "answers": {},
        "source_hashes": {
            str(p.relative_to(PROJECT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [PROJECT / "src", PROJECT.parent / "已有资源/可选组件/docparse/docparse"]
            for p in folder.rglob("*.py")
        },
    }

    def save():
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    try:
        for name in ["native", "scanned", "multi_scanned", "mixed_pages", "hybrid", "bad_layer"]:
            path = samples / (name + ".pdf")
            imported = service.ingest("pdf", str(path), {"mode": "page", "ocr_hybrid_images": True})
            if not imported.ok:
                raise RuntimeError(name + ": " + str(imported.as_dict()))
            doc = service.document(imported.documents[0]["doc_id"])
            result["documents"][name] = {
                "ingest": imported.as_dict(),
                "document": doc,
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            ir = doc["metadata"]["docparse_ir"]
            pages = ir["pages"]
            text = "\n".join(b.get("content", "") for p in pages for b in p["blocks"])
            compact = "".join(text.split())
            expected = {
                "native": 1,
                "scanned": 1,
                "multi_scanned": 2,
                "mixed_pages": 3,
                "hybrid": 1,
                "bad_layer": 1,
            }
            result["checks"][name + "_pages"] = len(pages) == expected[name]
            if name == "native":
                result["checks"]["native_route"] = (
                    all(p["mode"] == "native" for p in pages)
                    and "75.25" in compact
                    and all(b["source"] == "native_text" for p in pages for b in p["blocks"])
                )
            else:
                result["checks"][name + "_actual_purchase_ocr"] = all(
                    v in compact
                    for v in ["校园采购单", "SCAN-A1", "2026-11-25", "246.80", "93.5%", "PAID"]
                )
            if name == "multi_scanned":
                result["checks"]["second_page_fields"] = (
                    "每天24小时" in compact and "36months" in compact and pages[1]["number"] == 2
                )
            if name == "mixed_pages":
                result["checks"]["mixed_page_routes"] = [p["mode"] for p in pages] == [
                    "native",
                    "page",
                    "native",
                ]
            if name == "hybrid":
                result["checks"]["hybrid_native_preserved"] = "75.25" in text and any(
                    b["source"] == "native_text" for b in pages[0]["blocks"]
                )
                result["checks"]["hybrid_real_image_box"] = any(
                    b.get("meta", {}).get("bbox_scope") == "image_region_not_word_localization"
                    and b.get("bbox")
                    for b in pages[0]["blocks"]
                )
            if name == "bad_layer":
                result["checks"]["bad_text_layer_routed_to_ocr"] = (
                    pages[0]["kind"] == "scanned" and pages[0]["mode"] == "page"
                )
            if name in {"scanned", "multi_scanned", "bad_layer"}:
                result["checks"][name + "_no_fake_local_box"] = all(
                    not b.get("bbox")
                    for p in pages
                    for b in p["blocks"]
                    if b["source"] == "vlm_page"
                )
            save()
            print("PARSED", name, [p["kind"] + ":" + p["mode"] for p in pages], flush=True)
        batch = out / "image_batch"
        batch.mkdir()
        for file in ["bilingual.png", "bilingual.jpg"]:
            shutil.copyfile(samples / file, batch / file)
        (batch / "broken.png").write_bytes(b"broken image")
        report = service.ingest("connector.image", str(batch))
        result["image_batch"] = report.as_dict()
        result["checks"]["batch_two_success_one_failure"] = (
            len(report.documents) == 2
            and len(report.issues) == 1
            and report.issues[0].source == "broken.png"
        )
        documents = {d["source"]: d["doc_id"] for d in service.list_documents()}
        for file in ["bilingual.png", "bilingual.jpg"]:
            doc = service.document(documents[file])
            result["documents"][file] = doc
            text = "".join(doc["content"].split())
            result["checks"][file + "_bilingual_numeric_fields"] = all(
                v in text
                for v in ["校园采购单", "SCAN-A1", "246.80", "93.5%", "2026-11-25", "PAID"]
            )
        for key, question, source, values in [
            ("scan_answer", "采购单的金额是多少？", ["scanned.pdf"], ["246.80"]),
            ("image_answer", "What is the status in the receipt?", ["bilingual.png"], ["PAID"]),
            ("multi_answer", "设备保修期是多少个月？", ["multi_scanned.pdf"], ["36"]),
            (
                "hybrid_answer",
                "What is the native amount? What is the purchase amount?",
                ["hybrid.pdf"],
                ["75.25", "246.80"],
            ),
        ]:
            answer = service.ask(question, style="concise", doc_ids=[documents[s] for s in source])
            result["answers"][key] = answer
            result["checks"][key] = (
                answer["status"] in {"answered", "partial"}
                and all(v in answer["answer"] for v in values)
                and bool(answer["citations"])
                and all(c["source"] in source for c in answer["citations"])
            )
            save()
            print("ANSWER", key, answer["status"], answer["answer"], flush=True)
        unchanged = service.ingest("connector.image", str(batch))
        result["checks"]["repeated_batch_no_new_chunks"] = unchanged.ok and unchanged.chunks == 0
        # Author a changed input, then invoke the same OCR/ingestion path.
        subprocess.run(
            [
                PARSER,
                "-c",
                'from PIL import Image,ImageDraw,ImageFont; import sys; p=Image.open(sys.argv[1]); d=ImageDraw.Draw(p); d.rectangle((35,260,1080,345),fill="white"); d.text((50,269),"金额 Amount: 319.40 CNY",fill="black",font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",32)); p.save(sys.argv[1])',
                str(batch / "bilingual.png"),
            ],
            check=True,
        )
        changed = service.ingest("connector.image", str(batch / "bilingual.png"))
        result["changed_image_ingest"] = changed.as_dict()
        hits = service.search("采购单金额", doc_ids=[documents["bilingual.png"]])["hits"]
        result["changed_image_hits"] = hits
        result["checks"]["changed_actual_ocr_active_version"] = bool(hits) and all(
            h["version"] == 2 and "319.40" in h["text"] and "246.80" not in h["text"] for h in hits
        )
    except Exception as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)[:400]
        raise
    finally:
        service.close()
        result["status"] = (
            "PASS"
            if result["checks"] and all(result["checks"].values()) and not result.get("error")
            else "PARTIAL"
        )
        save()
    print("RESULT", result["status"], result["checks"], flush=True)


if __name__ == "__main__":
    main()
