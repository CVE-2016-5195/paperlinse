"""Vision LLM integration using Qwen3-VL with OpenVINO.

This module provides vision-based document analysis as an alternative to
text-based LLM processing. It uses Qwen3-VL through OpenVINO/Optimum Intel
for efficient inference on Intel hardware.

Supported models (pre-converted for OpenVINO):
- helenai/Qwen3-VL-2B-Instruct-int4 (~1.5GB) - recommended for most systems
- helenai/Qwen3-VL-4B-Instruct-int4 (~2.5GB) - better quality
- helenai/Qwen3-VL-8B-Instruct-int4 (~5GB) - best quality

Requirements:
    pip install -r requirements-vision.txt
    huggingface-cli download helenai/Qwen3-VL-2B-Instruct-int4 --local-dir models/Qwen3-VL-2B-Instruct-int4
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

from PIL import Image

from config import AppConfig
from llm import DocumentMetadata, DocumentIdentifierInfo

logger = logging.getLogger(__name__)

# Default model path - can be overridden by config
DEFAULT_VISION_MODEL_PATH = "models/Qwen3-VL-2B-Instruct-int4"
DEFAULT_VISION_DEVICE = "CPU"  # Can be "GPU", "CPU", or "NPU"

# Supported image formats that can be sent directly to the model
NATIVE_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png'}
# Formats that need conversion to JPEG before processing
CONVERT_IMAGE_FORMATS = {'.tiff', '.tif', '.bmp', '.gif', '.webp'}

# Maximum number of pages to process at once (to avoid OOM)
MAX_PAGES_PER_REQUEST = 10

# Default pixel limits for image resizing (can be overridden by config)
# Qwen2.5-VL uses 14x14 patches, dimensions should be multiples of 28
DEFAULT_MAX_PIXELS = 512 * 28 * 28  # ~401k pixels - balanced for iGPU INT4
DEFAULT_MIN_PIXELS = 4 * 28 * 28    # ~3.1k pixels - minimum size
PATCH_SIZE = 14
ROUND_TO = 28  # Dimensions should be multiples of this


def resize_image_for_vision(
    image: Image.Image,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_pixels: int = DEFAULT_MIN_PIXELS
) -> tuple[Image.Image, dict]:
    """
    Resize an image for optimal vision model processing.
    
    Only resizes if the image exceeds max_pixels. Preserves aspect ratio
    and rounds dimensions to multiples of 28 (required by Qwen2.5-VL).
    
    Args:
        image: PIL Image to potentially resize
        max_pixels: Maximum number of pixels (width * height)
        min_pixels: Minimum number of pixels (to ensure image isn't too small)
    
    Returns:
        Tuple of (resized_image, info_dict) where info_dict contains:
        - original_size: (width, height) of original image
        - new_size: (width, height) after resize (same as original if no resize)
        - resized: bool indicating if resize was performed
        - original_pixels: original pixel count
        - new_pixels: new pixel count
    """
    original_size = image.size
    original_pixels = original_size[0] * original_size[1]
    
    info = {
        "original_size": original_size,
        "new_size": original_size,
        "resized": False,
        "original_pixels": original_pixels,
        "new_pixels": original_pixels
    }
    
    # Check if resize is needed
    if original_pixels <= max_pixels:
        logger.debug(f"Image {original_size} ({original_pixels:,} px) within limit, no resize needed")
        return image, info
    
    # Calculate scale factor to fit within max_pixels
    scale = (max_pixels / original_pixels) ** 0.5
    
    # Calculate new dimensions
    new_width = int(original_size[0] * scale)
    new_height = int(original_size[1] * scale)
    
    # Round to multiples of ROUND_TO (28 for Qwen2.5-VL)
    new_width = max(ROUND_TO, (new_width // ROUND_TO) * ROUND_TO)
    new_height = max(ROUND_TO, (new_height // ROUND_TO) * ROUND_TO)
    
    # Ensure we don't go below min_pixels
    new_pixels = new_width * new_height
    if new_pixels < min_pixels:
        # Scale up to meet minimum
        min_scale = (min_pixels / new_pixels) ** 0.5
        new_width = max(ROUND_TO, int(new_width * min_scale // ROUND_TO) * ROUND_TO)
        new_height = max(ROUND_TO, int(new_height * min_scale // ROUND_TO) * ROUND_TO)
        new_pixels = new_width * new_height
    
    # Perform resize using LANCZOS for quality
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    info["new_size"] = (new_width, new_height)
    info["resized"] = True
    info["new_pixels"] = new_pixels
    
    logger.info(
        f"Resized image from {original_size[0]}x{original_size[1]} ({original_pixels:,} px) "
        f"to {new_width}x{new_height} ({new_pixels:,} px) "
        f"[{100 * new_pixels / original_pixels:.1f}% of original]"
    )
    
    return resized, info


@dataclass
class VisionLLMStatus:
    """Status of the Vision LLM."""
    available: bool
    model_path: str
    device: str
    error: Optional[str] = None


class VisionLLMClient:
    """Client for Qwen3-VL Vision LLM using OpenVINO."""
    
    _instance: Optional["VisionLLMClient"] = None
    
    def __init__(
        self,
        model_path: str = DEFAULT_VISION_MODEL_PATH,
        device: str = DEFAULT_VISION_DEVICE
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None
        self._initialized = False
        self._init_error: Optional[str] = None
    
    @classmethod
    def get_instance(cls) -> "VisionLLMClient":
        """Get singleton instance of Vision LLM client."""
        if cls._instance is None:
            config = AppConfig.load()
            model_path = config.vision_llm.model_path if hasattr(config, 'vision_llm') else DEFAULT_VISION_MODEL_PATH
            device = config.vision_llm.device if hasattr(config, 'vision_llm') else DEFAULT_VISION_DEVICE
            cls._instance = cls(model_path=model_path, device=device)
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for config changes)."""
        if cls._instance is not None:
            cls._instance._model = None
            cls._instance._processor = None
            cls._instance._initialized = False
        cls._instance = None
    
    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the Vision LLM model."""
        if self._initialized:
            return self._init_error is None
        
        self._initialized = True
        
        # Check if model path exists
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            self._init_error = f"Model not found at {self.model_path}. Please download with: huggingface-cli download helenai/Qwen3-VL-2B-Instruct-int4 --local-dir {self.model_path}"
            logger.error(self._init_error)
            return False
        
        try:
            # Force flush logs
            import sys
            print(f"[VISION_LLM] Loading Vision LLM from {self.model_path} on {self.device}...", file=sys.stderr, flush=True)
            logger.info(f"Loading Vision LLM from {self.model_path} on {self.device}...")
            
            # Import OpenVINO optimum intel
            from optimum.intel.openvino import OVModelForVisualCausalLM
            from transformers import AutoProcessor
            
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            print(f"[VISION_LLM] Processor loaded, now loading model on device={self.device}...", file=sys.stderr, flush=True)
            
            self._model = OVModelForVisualCausalLM.from_pretrained(
                self.model_path,
                device=self.device
            )
            
            # Log actual device info
            actual_device = getattr(self._model, '_device', 'unknown')
            print(f"[VISION_LLM] Model loaded. Model._device={actual_device}", file=sys.stderr, flush=True)
            logger.info(f"Vision LLM loaded successfully (device={actual_device})")
            return True
            
        except ImportError as e:
            self._init_error = f"OpenVINO dependencies not installed: {e}. Run: pip install -r requirements-vision.txt"
            logger.error(self._init_error)
            return False
        except Exception as e:
            self._init_error = f"Failed to load Vision LLM: {e}"
            logger.error(self._init_error)
            return False
    
    def is_available(self) -> bool:
        """Check if Vision LLM is available."""
        return self._ensure_initialized()
    
    def get_status(self) -> VisionLLMStatus:
        """Get the status of the Vision LLM."""
        self._ensure_initialized()
        return VisionLLMStatus(
            available=self._init_error is None,
            model_path=self.model_path,
            device=self.device,
            error=self._init_error
        )
    
    def convert_image_to_jpeg(self, image_path: str) -> str:
        """
        Convert an image to JPEG format if needed.
        
        Args:
            image_path: Path to the source image
        
        Returns:
            Path to the JPEG image (same path if already JPEG, temp file otherwise)
        """
        path = Path(image_path)
        suffix = path.suffix.lower()
        
        # Already a native format
        if suffix in NATIVE_IMAGE_FORMATS:
            return image_path
        
        # Need to convert
        if suffix in CONVERT_IMAGE_FORMATS:
            try:
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary (e.g., for TIFF with alpha)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix='.jpg',
                        delete=False,
                        prefix='paperlinse_vision_'
                    )
                    img.save(temp_file.name, 'JPEG', quality=95)
                    logger.debug(f"Converted {image_path} to {temp_file.name}")
                    return temp_file.name
            except Exception as e:
                logger.error(f"Failed to convert image {image_path}: {e}")
                raise
        
        raise ValueError(f"Unsupported image format: {suffix}")
    
    def extract_metadata(
        self,
        image_paths: list[str],
        language: str = "de"
    ) -> DocumentMetadata:
        """
        Extract document metadata from images using vision LLM.
        
        Args:
            image_paths: List of paths to document page images
            language: Language hint for extraction ('de' or 'en')
        
        Returns:
            DocumentMetadata with extracted fields
        """
        if not self._ensure_initialized():
            raise RuntimeError(self._init_error or "Vision LLM not initialized")
        
        if not image_paths:
            return DocumentMetadata(language=language)
        
        # Limit number of pages
        if len(image_paths) > MAX_PAGES_PER_REQUEST:
            logger.warning(
                f"Too many pages ({len(image_paths)}), "
                f"processing only first {MAX_PAGES_PER_REQUEST}"
            )
            image_paths = image_paths[:MAX_PAGES_PER_REQUEST]
        
        # Convert images if needed and prepare content
        converted_paths = []
        temp_files = []
        pil_images = []
        
        try:
            for img_path in image_paths:
                converted = self.convert_image_to_jpeg(img_path)
                converted_paths.append(converted)
                if converted != img_path:
                    temp_files.append(converted)
            
            # Build the prompt
            prompt = self._build_extraction_prompt(language, len(converted_paths))
            
            # Get pixel limits from config
            config = AppConfig.load()
            max_pixels = config.vision_llm.max_pixels if hasattr(config.vision_llm, 'max_pixels') else DEFAULT_MAX_PIXELS
            min_pixels = config.vision_llm.min_pixels if hasattr(config.vision_llm, 'min_pixels') else DEFAULT_MIN_PIXELS
            
            # Load images as PIL Image objects and resize if needed
            pil_images = []
            total_original_pixels = 0
            total_new_pixels = 0
            resize_count = 0
            
            for img_path in converted_paths:
                img = Image.open(img_path).convert("RGB")
                resized_img, resize_info = resize_image_for_vision(img, max_pixels, min_pixels)
                pil_images.append(resized_img)
                
                total_original_pixels += resize_info["original_pixels"]
                total_new_pixels += resize_info["new_pixels"]
                if resize_info["resized"]:
                    resize_count += 1
                
                # Close original if we created a new resized image
                if resize_info["resized"] and resized_img is not img:
                    img.close()
            
            # Log summary
            if resize_count > 0:
                logger.info(
                    f"Resized {resize_count}/{len(converted_paths)} images: "
                    f"{total_original_pixels:,} -> {total_new_pixels:,} total pixels "
                    f"({100 * total_new_pixels / total_original_pixels:.1f}%)"
                )
            
            # Log processing details
            import sys
            print(f"[VISION_LLM] Processing: model={self.model_path}, device={self.device}, images={len(pil_images)}, language={language}", file=sys.stderr, flush=True)
            logger.info(f"Vision LLM processing: model={self.model_path}, device={self.device}, images={len(pil_images)}, language={language}")
            
            # Build message content with images following Qwen2.5-VL format
            # Note: Qwen2.5-VL processor doesn't support system messages with image content,
            # so we include all instructions in the user message
            user_content = []
            for pil_img in pil_images:
                user_content.append({"type": "image", "image": pil_img})
            user_content.append({"type": "text", "text": prompt})
            
            # Build messages (no system message - Qwen2.5-VL processor has issues with it)
            messages = [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            # Process with model
            print(f"[VISION_LLM] Applying chat template...", file=sys.stderr, flush=True)
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            print(f"[VISION_LLM] Generating response (max_new_tokens=1024)...", file=sys.stderr, flush=True)
            logger.info(f"Generating metadata from {len(converted_paths)} image(s)...")
            
            # Get input length to strip from output later
            input_len = inputs["input_ids"].shape[1]
            
            generated_ids = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1024
            )
            
            print(f"[VISION_LLM] Decoding response...", file=sys.stderr, flush=True)
            
            # Only decode the newly generated tokens (strip input tokens)
            generated_ids_trimmed = generated_ids[:, input_len:]
            generated_texts = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )
            
            response = generated_texts[0] if generated_texts else ""
            print(f"[VISION_LLM] Response ({len(response)} chars): {response[:300]}...", file=sys.stderr, flush=True)
            logger.info(f"Vision LLM raw response ({len(response)} chars): {response[:500]}...")
            
            # Parse the response
            return self._parse_response(response, language)
            
        finally:
            # Clean up PIL images
            for pil_img in pil_images:
                try:
                    pil_img.close()
                except Exception:
                    pass
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
    
    def _build_extraction_prompt(self, language: str, num_pages: int) -> str:
        """Build the extraction prompt based on language, using config prompts."""
        from config import AppConfig, DEFAULT_VISION_PROMPT_DE, DEFAULT_VISION_PROMPT_EN
        
        # Load prompts from config
        config = AppConfig.load()
        
        if language == "de":
            prompt = getattr(config.vision_llm, 'prompt_de', None) or DEFAULT_VISION_PROMPT_DE
        else:
            prompt = getattr(config.vision_llm, 'prompt_en', None) or DEFAULT_VISION_PROMPT_EN
        
        # Add page context if multiple pages
        if num_pages > 1:
            if language == "de":
                page_context = f"\n\nDieses Dokument hat {num_pages} Seiten (oben gezeigt)."
            else:
                page_context = f"\n\nThis document has {num_pages} pages shown above."
            prompt = prompt + page_context
        
        return prompt
    
    def _parse_response(self, response: str, language: str) -> DocumentMetadata:
        """Parse the LLM response into DocumentMetadata."""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, c in enumerate(response[start_idx:], start_idx):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                data = json.loads(response)
            
            # Parse document_date
            doc_date = None
            if data.get("document_date"):
                try:
                    doc_date = datetime.strptime(data["document_date"], "%Y-%m-%d")
                except ValueError:
                    pass
            
            # Parse due_date
            due_date = None
            if data.get("due_date"):
                try:
                    due_date = datetime.strptime(data["due_date"], "%Y-%m-%d")
                except ValueError:
                    pass
            
            # Parse identifiers
            identifiers = []
            raw_identifiers = data.get("identifiers", [])
            if isinstance(raw_identifiers, list):
                for ident in raw_identifiers:
                    if isinstance(ident, dict):
                        ident_type = ident.get("type", "").strip()
                        ident_value = ident.get("value", "").strip()
                        if ident_type and ident_value:
                            identifiers.append(DocumentIdentifierInfo(
                                type=ident_type[:100],
                                value=ident_value[:500]
                            ))
            
            return DocumentMetadata(
                sender=data.get("sender", "")[:500],
                receiver=data.get("receiver", "")[:500],
                document_date=doc_date,
                topic=data.get("topic", "")[:500],
                summary=data.get("summary", "")[:1000],
                document_type=data.get("document_type", "other"),
                language=language,
                confidence=0.85,  # Vision typically has good confidence
                identifiers=identifiers,
                iban=data.get("iban", "")[:50] if data.get("iban") else "",
                bic=data.get("bic", "")[:20] if data.get("bic") else "",
                due_date=due_date,
                raw_response=data
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Vision LLM response as JSON: {e}")
            logger.debug(f"Response was: {response}")
            
            return DocumentMetadata(
                language=language,
                confidence=0.3
            )


# Global functions for easy access

_vision_client: Optional[VisionLLMClient] = None


def get_vision_llm_client() -> VisionLLMClient:
    """Get the global Vision LLM client instance."""
    global _vision_client
    if _vision_client is None:
        _vision_client = VisionLLMClient.get_instance()
    return _vision_client


def is_vision_llm_available() -> bool:
    """Check if Vision LLM is available and initialized."""
    try:
        client = get_vision_llm_client()
        return client.is_available()
    except Exception:
        return False


def get_vision_llm_status() -> dict:
    """Get the status of the Vision LLM as a dictionary."""
    try:
        client = get_vision_llm_client()
        status = client.get_status()
        return {
            "available": status.available,
            "model_path": status.model_path,
            "device": status.device,
            "error": status.error
        }
    except Exception as e:
        return {
            "available": False,
            "model_path": DEFAULT_VISION_MODEL_PATH,
            "device": DEFAULT_VISION_DEVICE,
            "error": str(e)
        }


async def extract_metadata_vision(
    image_paths: list[str],
    language: str = "de"
) -> DocumentMetadata:
    """
    Extract document metadata using vision LLM.
    
    This is an async wrapper for use in the processing pipeline.
    """
    import asyncio
    
    client = get_vision_llm_client()
    
    # Run in thread pool since OpenVINO inference is synchronous
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: client.extract_metadata(image_paths, language)
    )
