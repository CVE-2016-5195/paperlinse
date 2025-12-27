"""Ollama LLM integration for document analysis.

Supports both standard Ollama and IPEX-LLM Ollama for Intel GPU acceleration.
When using Intel GPU mode (--intel-gpu flag), recommended models include:
- qwen3:0.6b (default for Intel GPU, optimized for low-power devices)
- qwen3:1.7b (better quality, requires more GPU memory)
- llama3.2:1b (alternative option)

Configure via environment variables:
- OLLAMA_BASE_URL: Ollama API endpoint (default: http://localhost:11434)
- OLLAMA_MODEL: Model to use (default: qwen2.5:1.5b, or qwen3:0.6b for Intel GPU)
"""

import os
import json
import logging
import re
import asyncio
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from config import AppConfig

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")


@dataclass
class DocumentIdentifierInfo:
    """A detected identifier in a document."""
    type: str = ""   # e.g., "Vorgangsnummer", "Rechnungsnummer", "Aktenzeichen"
    value: str = ""


@dataclass
class DocumentMetadata:
    """Extracted metadata from a document."""
    sender: str = ""
    receiver: str = ""
    document_date: Optional[datetime] = None
    topic: str = ""
    summary: str = ""
    document_type: str = ""  # invoice, letter, contract, etc.
    language: str = ""
    page_info: Optional[str] = None  # e.g., "Page 1 of 3"
    confidence: float = 0.0
    # Enhanced metadata
    identifiers: list[DocumentIdentifierInfo] = field(default_factory=list)
    iban: str = ""
    bic: str = ""
    due_date: Optional[datetime] = None
    raw_response: Optional[dict] = None  # Complete LLM response for debugging


@dataclass
class PageGroupingResult:
    """Result of page grouping analysis."""
    belongs_to_same_document: bool
    confidence: float
    reason: str


class OllamaClient:
    """Client for Ollama LLM API."""
    
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = 120.0
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """Generate text using Ollama."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            if system:
                payload["system"] = system
            
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            elif response.status_code == 404:
                # Model not found - provide helpful error message
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("error", "Model not found")
                logger.error(f"Ollama model not found: {self.model} - {error_msg}")
                raise ValueError(f"LLM model '{self.model}' not found. Please check available models with 'ollama list' and update OLLAMA_MODEL environment variable.")
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                raise RuntimeError(f"LLM error ({response.status_code}): {response.text[:200]}")
                
        except httpx.TimeoutException:
            logger.error(f"Ollama timeout - model may be loading")
            raise RuntimeError("LLM timeout - model may still be loading, please try again")
        except ValueError:
            raise  # Re-raise ValueError (model not found)
        except RuntimeError:
            raise  # Re-raise RuntimeError
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"LLM error: {str(e)}")
    
    async def extract_metadata(self, ocr_text: str, language: str = "de") -> DocumentMetadata:
        """
        Extract structured metadata from OCR text using LLM.
        
        Args:
            ocr_text: The OCR-extracted text from the document
            language: Detected language of the document
        
        Returns:
            DocumentMetadata with extracted fields
        """
        if not ocr_text.strip():
            return DocumentMetadata(language=language)
        
        # Truncate very long texts
        max_chars = 4000
        text_for_analysis = ocr_text[:max_chars] if len(ocr_text) > max_chars else ocr_text
        
        # Load prompts from config
        config = AppConfig.load()
        system_prompt = config.llm.system_prompt
        
        # Select and format user prompt based on language
        if language == "de":
            user_prompt = config.llm.user_prompt_de.format(text=text_for_analysis)
        else:
            user_prompt = config.llm.user_prompt_en.format(text=text_for_analysis)

        response = await self.generate(user_prompt, system_prompt)
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response - handle nested objects
            # Find the outermost braces
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
                page_info=data.get("page_info", ""),
                confidence=0.8,  # Assume reasonable confidence for successful extraction
                identifiers=identifiers,
                iban=data.get("iban", "")[:50] if data.get("iban") else "",
                bic=data.get("bic", "")[:20] if data.get("bic") else "",
                due_date=due_date,
                raw_response=data
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response}")
            
            # Return minimal metadata
            return DocumentMetadata(
                language=language,
                confidence=0.3
            )
    
    async def should_group_pages(
        self,
        page1_text: str,
        page2_text: str,
        page1_metadata: DocumentMetadata,
        page2_metadata: DocumentMetadata
    ) -> PageGroupingResult:
        """
        Determine if two pages belong to the same document.
        
        Uses multiple signals:
        - Same sender
        - Same topic/subject
        - Page numbering (e.g., "Page 1/3", "Page 2/3")
        - Content continuity
        """
        # Quick checks first (no LLM needed)
        
        # Check for explicit page numbering
        page1_num = self._extract_page_number(page1_metadata.page_info or page1_text)
        page2_num = self._extract_page_number(page2_metadata.page_info or page2_text)
        
        if page1_num and page2_num:
            if page1_num[1] == page2_num[1]:  # Same total pages
                if page2_num[0] == page1_num[0] + 1:  # Sequential
                    return PageGroupingResult(
                        belongs_to_same_document=True,
                        confidence=0.95,
                        reason=f"Sequential page numbers: {page1_num[0]}/{page1_num[1]} -> {page2_num[0]}/{page2_num[1]}"
                    )
        
        # Check if sender matches
        if (page1_metadata.sender and page2_metadata.sender and 
            page1_metadata.sender.lower() == page2_metadata.sender.lower()):
            # Same sender - likely same document, but ask LLM for confirmation
            pass
        
        # Use LLM for more complex analysis
        system_prompt = """You are a document analyst. Determine if two pages belong to the same document.
Consider: sender/letterhead, topic continuity, page numbers, date, formatting consistency.
Respond with JSON only."""

        prompt = f"""Do these two pages belong to the SAME document?

PAGE 1:
{page1_text[:1500]}

PAGE 2:
{page2_text[:1500]}

Respond with JSON:
{{
    "same_document": true or false,
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}}"""

        response = await self.generate(prompt, system_prompt)
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return PageGroupingResult(
                belongs_to_same_document=data.get("same_document", False),
                confidence=data.get("confidence", 0.5),
                reason=data.get("reason", "")
            )
            
        except json.JSONDecodeError:
            # Default to not grouping if uncertain
            return PageGroupingResult(
                belongs_to_same_document=False,
                confidence=0.3,
                reason="Could not determine relationship"
            )
    
    def _extract_page_number(self, text: str) -> Optional[tuple[int, int]]:
        """
        Extract page number info like "Page 1 of 3" or "Seite 2/5".
        
        Returns:
            Tuple of (current_page, total_pages) or None
        """
        if not text:
            return None
        
        patterns = [
            r'[Pp]age\s*(\d+)\s*(?:of|/)\s*(\d+)',
            r'[Ss]eite\s*(\d+)\s*(?:von|/)\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)',
            r'-\s*(\d+)\s*-.*?(\d+)\s*[Ss]eiten?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    if 0 < current <= total <= 100:  # Sanity check
                        return (current, total)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    async def generate_folder_name(
        self,
        metadata: DocumentMetadata,
        max_length: int = 80
    ) -> str:
        """
        Generate a descriptive folder name for the document.
        
        Format: YYYY-MM-DD_Topic-Sender
        """
        # Date prefix
        if metadata.document_date:
            date_str = metadata.document_date.strftime("%Y-%m-%d")
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Sanitize topic for filesystem
        topic = metadata.topic or metadata.document_type or "Document"
        topic = self._sanitize_filename(topic)
        
        # Add sender if available
        sender = ""
        if metadata.sender:
            sender = self._sanitize_filename(metadata.sender)
            sender = f"-{sender}"
        
        # Build folder name
        folder_name = f"{date_str}_{topic}{sender}"
        
        # Truncate if too long
        if len(folder_name) > max_length:
            folder_name = folder_name[:max_length].rstrip("-_")
        
        return folder_name
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use in filenames."""
        # Replace problematic characters
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'\s+', '-', name)
        name = re.sub(r'-+', '-', name)
        name = name.strip('-_. ')
        
        # Limit length of individual component
        if len(name) > 50:
            name = name[:50].rstrip('-_')
        
        return name


# Singleton instance
_client: Optional[OllamaClient] = None

# Model pull progress tracking
_model_pull_status: dict = {}

# Default model if none configured
DEFAULT_MODEL = "qwen3:0.6b"


async def get_ollama_client() -> OllamaClient:
    """Get the global Ollama client instance, using model from config if set."""
    global _client
    if _client is None:
        # Load model from config, fallback to default
        config = AppConfig.load()
        model = config.llm.model if config.llm.model else DEFAULT_MODEL
        _client = OllamaClient(model=model)
    return _client


def get_current_model() -> str:
    """Get the currently configured LLM model name."""
    config = AppConfig.load()
    return config.llm.model if config.llm.model else DEFAULT_MODEL


async def set_current_model(model: str) -> None:
    """Set the current LLM model and update the global client."""
    global _client
    config = AppConfig.load()
    config.llm.model = model
    config.save()
    
    # Update or recreate the client with new model
    if _client is not None:
        _client.model = model
    else:
        _client = OllamaClient(model=model)


async def list_available_models() -> list[dict]:
    """List all models available in Ollama."""
    client = await get_ollama_client()
    try:
        response = await client._client.get(f"{client.base_url}/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "modified_at": m.get("modified_at", ""),
                    "digest": m.get("digest", "")[:12] if m.get("digest") else ""
                })
            return models
        return []
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []


async def is_model_available(model: str) -> bool:
    """Check if a specific model is available locally."""
    models = await list_available_models()
    return any(m["name"] == model or m["name"].startswith(f"{model}:") for m in models)


async def pull_model(model: str) -> AsyncGenerator[dict, None]:
    """
    Pull/download a model from Ollama registry.
    Yields progress updates as they come in.
    """
    global _model_pull_status
    
    client = await get_ollama_client()
    _model_pull_status[model] = {"status": "starting", "progress": 0, "total": 0}
    
    try:
        async with client._client.stream(
            "POST",
            f"{client.base_url}/api/pull",
            json={"name": model, "stream": True},
            timeout=600.0  # 10 minute timeout for large models
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    
                    # Parse progress info
                    progress_info = {
                        "status": status,
                        "completed": data.get("completed", 0),
                        "total": data.get("total", 0),
                        "digest": data.get("digest", "")[:12] if data.get("digest") else ""
                    }
                    
                    # Calculate percentage
                    if progress_info["total"] > 0:
                        progress_info["percent"] = int(
                            (progress_info["completed"] / progress_info["total"]) * 100
                        )
                    else:
                        progress_info["percent"] = 0
                    
                    _model_pull_status[model] = progress_info
                    yield progress_info
                    
                except json.JSONDecodeError:
                    continue
        
        # Mark as complete
        _model_pull_status[model] = {"status": "success", "percent": 100}
        yield {"status": "success", "percent": 100}
        
    except Exception as e:
        logger.error(f"Error pulling model {model}: {e}")
        _model_pull_status[model] = {"status": "error", "error": str(e)}
        yield {"status": "error", "error": str(e)}
    finally:
        # Clean up after a delay
        await asyncio.sleep(5)
        if model in _model_pull_status:
            del _model_pull_status[model]


def get_model_pull_status(model: str) -> Optional[dict]:
    """Get the current pull status for a model."""
    return _model_pull_status.get(model)


async def delete_model(model: str) -> bool:
    """Delete a model from Ollama."""
    client = await get_ollama_client()
    try:
        response = await client._client.delete(
            f"{client.base_url}/api/delete",
            json={"name": model}
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error deleting model {model}: {e}")
        return False


async def extract_document_metadata(ocr_text: str, language: str = "de") -> DocumentMetadata:
    """Convenience function to extract metadata from OCR text."""
    client = await get_ollama_client()
    return await client.extract_metadata(ocr_text, language)
