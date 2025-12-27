"""PaddleOCR wrapper for document text extraction."""

import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Disable oneDNN for PaddlePaddle to avoid conflicts with Intel GPU compute runtime
# This prevents "could not create an engine" errors when IPEX-LLM Ollama is running
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["DNNL_VERBOSE"] = "0"

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    language: str
    boxes: list[dict]  # Bounding boxes with text and confidence


class PaddleOCREngine:
    """PaddleOCR wrapper for document text extraction."""
    
    _instance: Optional["PaddleOCREngine"] = None
    
    def __init__(self):
        """Initialize PaddleOCR engine."""
        self._ocr_de = None
        self._ocr_en = None
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> "PaddleOCREngine":
        """Get singleton instance of OCR engine."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensure_initialized(self, lang: str = "de") -> None:
        """Lazy initialization of PaddleOCR models."""
        if lang == "de" and self._ocr_de is None:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR for German...")
            self._ocr_de = PaddleOCR(
                use_angle_cls=True,
                lang='german',
                use_gpu=self._check_gpu_available(),
                show_log=False,
                # Enable document structure analysis
                structure_version='PP-StructureV2',
            )
            logger.info("PaddleOCR German model initialized")
        
        elif lang == "en" and self._ocr_en is None:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR for English...")
            self._ocr_en = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self._check_gpu_available(),
                show_log=False,
                structure_version='PP-StructureV2',
            )
            logger.info("PaddleOCR English model initialized")
        
        self._initialized = True
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for inference."""
        try:
            import paddle
            return paddle.device.is_compiled_with_cuda()
        except Exception:
            return False
    
    def _get_ocr(self, lang: str):
        """Get the appropriate OCR model for the language."""
        self._ensure_initialized(lang)
        if lang == "en":
            return self._ocr_en
        return self._ocr_de  # Default to German
    
    def extract_text(
        self,
        image_path: str,
        lang: str = "de"
    ) -> OCRResult:
        """
        Extract text from an image using PaddleOCR.
        
        Args:
            image_path: Path to the image file
            lang: Language code ('de' for German, 'en' for English)
        
        Returns:
            OCRResult with extracted text and metadata
        """
        try:
            ocr = self._get_ocr(lang)
            
            # Run OCR
            result = ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    language=lang,
                    boxes=[]
                )
            
            # Process results
            lines = []
            boxes = []
            total_confidence = 0.0
            
            for line in result[0]:
                if not line:
                    continue
                
                # Handle case where line might be a simple string or unexpected format
                if isinstance(line, str):
                    lines.append(line)
                    boxes.append({
                        "box": [],
                        "text": line,
                        "confidence": 0.5
                    })
                    total_confidence += 0.5
                    continue
                
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    logger.warning(f"Unexpected OCR line structure: {type(line)}")
                    continue
                    
                box_coords = line[0]
                text_info = line[1]
                
                # Handle various formats PaddleOCR might return
                text = ""
                confidence = 0.0
                
                if isinstance(text_info, tuple) and len(text_info) >= 2:
                    text = str(text_info[0]) if text_info[0] else ""
                    confidence = float(text_info[1]) if isinstance(text_info[1], (int, float)) else 0.0
                elif isinstance(text_info, list) and len(text_info) >= 2:
                    text = str(text_info[0]) if text_info[0] else ""
                    confidence = float(text_info[1]) if isinstance(text_info[1], (int, float)) else 0.0
                elif isinstance(text_info, str):
                    text = text_info
                    confidence = 0.5  # Unknown confidence
                elif isinstance(text_info, dict):
                    text = str(text_info.get('text', ''))
                    confidence = float(text_info.get('confidence', 0.0))
                elif isinstance(text_info, (int, float)):
                    # Sometimes OCR returns just a number, skip these
                    logger.debug(f"Skipping numeric OCR result: {text_info}")
                    continue
                else:
                    # Skip if we can't parse this
                    logger.warning(f"Unexpected OCR text_info format: {type(text_info)} - {text_info}")
                    continue
                
                if not text:
                    continue
                
                lines.append(text)
                total_confidence += confidence
                
                boxes.append({
                    "box": box_coords,
                    "text": text,
                    "confidence": confidence
                })
            
            full_text = "\n".join(lines)
            avg_confidence = total_confidence / len(boxes) if boxes else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                language=lang,
                boxes=boxes
            )
            
        except Exception as e:
            logger.error(f"OCR error for {image_path}: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                language=lang,
                boxes=[]
            )
    
    def extract_text_from_pdf(
        self,
        pdf_path: str,
        lang: str = "de",
        dpi: int = 200
    ) -> list[OCRResult]:
        """
        Extract text from all pages of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            lang: Language code
            dpi: Resolution for PDF to image conversion
        
        Returns:
            List of OCRResult, one per page
        """
        try:
            import pdf2image
            
            # Convert PDF pages to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt='jpeg'
            )
            
            results = []
            for i, image in enumerate(images):
                # Save temp image
                temp_path = f"/tmp/paperlinse_pdf_page_{i}.jpg"
                image.save(temp_path, "JPEG", quality=95)
                
                # OCR the page
                result = self.extract_text(temp_path, lang)
                results.append(result)
                
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
            return results
            
        except Exception as e:
            logger.error(f"PDF OCR error for {pdf_path}: {e}")
            return []
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily German or English.
        
        Uses simple heuristics based on common words.
        """
        if not text:
            return "de"  # Default to German
        
        text_lower = text.lower()
        
        # German indicators
        german_words = [
            'und', 'der', 'die', 'das', 'ist', 'für', 'von', 'mit',
            'auf', 'den', 'ein', 'eine', 'nicht', 'sich', 'auch',
            'werden', 'bei', 'können', 'haben', 'wird', 'sind',
            'sehr', 'geehrte', 'rechnung', 'betreff', 'datum',
            'straße', 'strasse', 'gmbh', 'telefon', 'bitte'
        ]
        
        # English indicators
        english_words = [
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'her', 'was', 'one', 'our', 'out', 'has', 'his',
            'have', 'from', 'they', 'been', 'would', 'their',
            'invoice', 'dear', 'regarding', 'please', 'thank'
        ]
        
        german_count = sum(1 for word in german_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        # Check for German special characters (strong indicator)
        german_chars = ['ä', 'ö', 'ü', 'ß']
        if any(char in text_lower for char in german_chars):
            german_count += 5
        
        return "de" if german_count >= english_count else "en"
    
    def process_document(
        self,
        file_path: str,
        auto_detect_lang: bool = True
    ) -> list[OCRResult]:
        """
        Process a document file (image or PDF) with OCR.
        
        Args:
            file_path: Path to the document file
            auto_detect_lang: Whether to auto-detect language
        
        Returns:
            List of OCRResult, one per page
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        # Start with German (most common in this use case)
        initial_lang = "de"
        
        if suffix == '.pdf':
            results = self.extract_text_from_pdf(file_path, initial_lang)
        elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']:
            results = [self.extract_text(file_path, initial_lang)]
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []
        
        # Auto-detect language and re-run if needed
        if auto_detect_lang and results:
            combined_text = " ".join(r.text for r in results)
            detected_lang = self.detect_language(combined_text)
            
            if detected_lang != initial_lang:
                logger.info(f"Re-running OCR with detected language: {detected_lang}")
                if suffix == '.pdf':
                    results = self.extract_text_from_pdf(file_path, detected_lang)
                else:
                    results = [self.extract_text(file_path, detected_lang)]
        
        return results


# Convenience function
def process_document(file_path: str) -> list[OCRResult]:
    """Process a document with OCR."""
    engine = PaddleOCREngine.get_instance()
    return engine.process_document(file_path)
