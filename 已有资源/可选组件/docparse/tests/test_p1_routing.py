"""Deterministic page routing and failure isolation; OCR here is a test double."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz
from docparse.config import ParseOptions
from docparse.ir import Block
from docparse.pipeline import Parser
from PIL import Image


class TestP1Routing(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        Image.new('RGB',(300,400),'white').save(self.root/'scan.png')

    def tearDown(self):
        self.temp.cleanup()

    def pdf(self,hybrid=False,pages=1):
        document=fitz.open()
        for i in range(pages):
            page=document.new_page(width=300,height=400)
            page.insert_image(page.rect,filename=str(self.root/'scan.png'))
            if hybrid:
                page.insert_text((20,30),'Healthy native text preserved and never sent to OCR. Native 75.25 CNY.',fontsize=7)
        path=self.root/'input.pdf'
        document.save(path)
        document.close()
        return path

    @staticmethod
    def ocr(image,number):
        return [Block(id='ocr',page=number,type='page',content='Receipt 246.80 CNY',source='vlm_page')],'page',[],1,{}

    def test_hybrid_opt_in_keeps_native_and_measured_image_region(self):
        parser=Parser(ParseOptions(mode='page',ocr_hybrid_images=True))
        with patch.object(parser,'_ocr',side_effect=self.ocr) as call:
            document=parser.parse(self.pdf(True))
        self.assertEqual(call.call_count,1)
        blocks=document.pages[0].blocks
        self.assertTrue(any('75.25' in b.content and b.source=='native_text' for b in blocks))
        ocr=next(b for b in blocks if '246.80' in b.content)
        self.assertEqual(ocr.bbox,[0.0,0.0,1000.0,1000.0])
        self.assertEqual(ocr.meta['bbox_scope'],'image_region_not_word_localization')
        self.assertEqual(ocr.type,'page_text')

    def test_hybrid_default_does_not_silently_claim_image_text(self):
        parser=Parser(ParseOptions(mode='page',ocr_hybrid_images=False))
        with patch.object(parser,'_ocr') as call:
            document=parser.parse(self.pdf(True))
        call.assert_not_called()
        self.assertTrue(any(w['code']=='hybrid_images_not_ocr' for w in document.pages[0].warnings))

    def test_failed_scan_page_does_not_erase_successful_other_page(self):
        parser=Parser(ParseOptions(mode='page'))
        with patch.object(parser,'_ocr',side_effect=[RuntimeError('one failed page'),self.ocr(None,2)]):
            document=parser.parse(self.pdf(pages=2))
        self.assertEqual([p.mode for p in document.pages],['failed','page'])
        self.assertIn('246.80',document.pages[1].blocks[0].content)
        self.assertTrue(document.needs_review)
        self.assertEqual(document.warnings[0]['page'],1)

    def test_render_failure_is_isolated_before_ocr(self):
        parser=Parser(ParseOptions(mode='page'))
        with patch('docparse.pipeline.pdfio.render_page',side_effect=[RuntimeError('render failed'),Image.new('RGB',(300,400),'white')]), patch.object(parser,'_ocr',side_effect=self.ocr) as ocr:
            document=parser.parse(self.pdf(pages=2))
        self.assertEqual([p.mode for p in document.pages],['failed','page'])
        self.assertEqual(ocr.call_count,1)
        self.assertEqual(document.warnings[0]['code'],'page_failed')
        self.assertEqual(document.warnings[0]['page'],1)
        self.assertIn('246.80',document.pages[1].blocks[0].content)

    def test_whole_page_ocr_does_not_manufacture_local_box(self):
        parser=Parser(ParseOptions(mode='page'))
        with patch('docparse.pipeline.backend.chat',return_value={'text':'Receipt amount: 246.80 CNY','finish_reason':'stop','usage':{}}):
            blocks,_,_=parser._ocr_page(Image.new('RGB',(300,400),'white'),1)
        self.assertEqual(blocks[0].bbox,[])
        self.assertEqual(blocks[0].meta['bbox_scope'],'unavailable')

    def test_unknown_backend_failure_only_discards_one_embedded_image(self):
        class OptionalBackendFailure(Exception):
            pass
        document = fitz.open()
        page = document.new_page(width=300, height=400)
        page.insert_text((20, 30), 'Healthy native text preserved with two separate embedded images.', fontsize=7)
        page.insert_image(fitz.Rect(10, 60, 140, 380), filename=str(self.root/'scan.png'))
        page.insert_image(fitz.Rect(160, 60, 290, 380), filename=str(self.root/'scan.png'))
        path = self.root/'two_images.pdf'
        document.save(path)
        document.close()
        parser = Parser(ParseOptions(mode='page', ocr_hybrid_images=True))
        with patch.object(parser, '_ocr', side_effect=[OptionalBackendFailure('backend mismatch'), self.ocr(None, 1)]) as call:
            parsed = parser.parse(path)
        self.assertEqual(call.call_count, 2)
        self.assertTrue(any(w['code'] == 'hybrid_image_failed' for w in parsed.pages[0].warnings))
        self.assertTrue(any('246.80' in b.content for b in parsed.pages[0].blocks))
        self.assertTrue(any('Healthy native' in b.content for b in parsed.pages[0].blocks))


if __name__=='__main__':
    unittest.main()
