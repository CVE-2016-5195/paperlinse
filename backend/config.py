"""Configuration management for Paperlinse."""

import os
import json
import base64
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_FILE = CONFIG_DIR / "settings.json"
KEY_FILE = CONFIG_DIR / ".key"


def get_or_create_key() -> bytes:
    """Get or create encryption key for sensitive data."""
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_bytes(key)
    KEY_FILE.chmod(0o600)
    return key


def encrypt_value(value: str) -> str:
    """Encrypt a sensitive value."""
    if not value:
        return ""
    f = Fernet(get_or_create_key())
    return base64.urlsafe_b64encode(f.encrypt(value.encode())).decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt a sensitive value."""
    if not encrypted:
        return ""
    try:
        f = Fernet(get_or_create_key())
        return f.decrypt(base64.urlsafe_b64decode(encrypted.encode())).decode()
    except Exception:
        return ""


class ShareCredentials(BaseModel):
    """Credentials for network share access."""
    username: str = ""
    password: str = ""  # Stored encrypted
    domain: str = ""


class IncomingShareConfig(BaseModel):
    """Configuration for the incoming document share."""
    path: str = ""
    credentials: ShareCredentials = Field(default_factory=ShareCredentials)
    poll_interval_seconds: int = 30


class StorageConfig(BaseModel):
    """Configuration for document storage."""
    path: str = ""
    credentials: ShareCredentials = Field(default_factory=ShareCredentials)


# Default LLM prompts - optimized for document classification
DEFAULT_SYSTEM_PROMPT = """You are a document analysis assistant. Extract structured information from documents.
Always respond in valid JSON format only, no other text.
For dates, use ISO format (YYYY-MM-DD).
If a field cannot be determined, use an empty string or empty array as appropriate."""

DEFAULT_USER_PROMPT_DE = """Analysiere dieses Dokument und extrahiere die folgenden Informationen.

WICHTIG für das "topic" Feld:
1. Suche nach der HAUPTÜBERSCHRIFT oder dem BETREFF des Dokuments
2. Bei offiziellen Briefen: Suche nach "Betreff:", "Betrifft:", oder einer fett gedruckten Überschrift
3. Bei Formularen: Nutze den Titel des Formulars (z.B. "Vorladung", "Bescheid", "Mitteilung")
4. Der topic sollte den GRUND oder ZWECK des Dokuments beschreiben, nicht nur die Dokumentart
5. Beispiele: "Vorladung zur Vernehmung", "Kündigung Arbeitsvertrag", "Mahnung Rechnung 12345", "Steuerbescheid 2023"

WICHTIG für "identifiers" (Kennzeichen/Referenznummern):
1. Suche nach ALLEN Referenznummern, Aktenzeichen, Vorgangsnummern, Kundennummern, Rechnungsnummern, usw.
2. Beispiele für Kennzeichentypen: Vorgangsnummer, Aktenzeichen, Rechnungsnummer, Kundennummer, Auftragsnummer, Vertragsnummer, Fallnummer, Buchungsnummer
3. Jedes Kennzeichen hat einen "type" (was es ist) und einen "value" (der Wert)

DOKUMENT:
{text}

Antworte NUR mit einem JSON-Objekt in diesem Format:
{{
    "sender": "Name/Organisation des Absenders (z.B. Firma, Behörde, Person)",
    "receiver": "Name des Empfängers falls erkennbar",
    "document_date": "YYYY-MM-DD (Datum des Dokuments, nicht Eingangsdatum)",
    "topic": "Hauptthema/Betreff/Grund des Dokuments (aus Überschrift oder Betreffzeile)",
    "summary": "Kurze Zusammenfassung des Inhalts (max 200 Zeichen)",
    "document_type": "letter|invoice|payslip|contract|receipt|statement|notification|summons|certificate|form|report|other",
    "page_info": "z.B. 'Seite 1 von 3' wenn vorhanden",
    "identifiers": [
        {{"type": "Kennzeichenart", "value": "Wert"}},
        {{"type": "z.B. Rechnungsnummer", "value": "z.B. RE-2024-12345"}}
    ],
    "iban": "IBAN falls vorhanden (z.B. DE89370400440532013000)",
    "bic": "BIC/SWIFT falls vorhanden",
    "due_date": "YYYY-MM-DD (Fälligkeitsdatum bei Rechnungen/Zahlungen)"
}}"""

DEFAULT_USER_PROMPT_EN = """Analyze this document and extract the following information.

IMPORTANT for the "topic" field:
1. Look for the MAIN HEADING or SUBJECT of the document
2. For official letters: Look for "Subject:", "Re:", or a bold heading
3. For forms: Use the form title (e.g. "Summons", "Notice", "Certificate")
4. The topic should describe the PURPOSE or REASON of the document, not just the document type
5. Examples: "Summons for Questioning", "Employment Contract Termination", "Payment Reminder Invoice 12345", "Tax Assessment 2023"

IMPORTANT for "identifiers" (reference numbers):
1. Look for ALL reference numbers, case numbers, transaction numbers, customer numbers, invoice numbers, etc.
2. Examples of identifier types: Reference Number, Case Number, Invoice Number, Customer Number, Order Number, Contract Number, Booking Number
3. Each identifier has a "type" (what it is) and a "value" (the actual number/code)

DOCUMENT:
{text}

Respond ONLY with a JSON object in this format:
{{
    "sender": "Name/Organization of sender (e.g. company, authority, person)",
    "receiver": "Name of recipient if identifiable",
    "document_date": "YYYY-MM-DD (date of document, not receipt date)",
    "topic": "Main topic/subject/reason of document (from heading or subject line)",
    "summary": "Brief summary of content (max 200 chars)",
    "document_type": "letter|invoice|payslip|contract|receipt|statement|notification|summons|certificate|form|report|other",
    "page_info": "e.g. 'Page 1 of 3' if present",
    "identifiers": [
        {{"type": "Identifier type", "value": "Value"}},
        {{"type": "e.g. Invoice Number", "value": "e.g. INV-2024-12345"}}
    ],
    "iban": "IBAN if present (e.g. DE89370400440532013000)",
    "bic": "BIC/SWIFT if present",
    "due_date": "YYYY-MM-DD (due date for invoices/payments)"
}}"""


# Recommended LLM models - all multilingual with German support
# Format: (model_id, display_name, size_mb, description, model_type)
# model_type: "text" for Ollama models, "vision" for OpenVINO vision models
RECOMMENDED_MODELS = [
    ("granite4:350m", "Granite4 350M", 512, "IBM, 12 languages, smallest & fastest", "text"),
    ("qwen2.5:0.5b", "Qwen2.5 0.5B", 600, "29 languages, good JSON output", "text"),
    ("qwen3:0.6b", "Qwen3 0.6B", 768, "100+ languages, good for low-power devices", "text"),
    ("gemma3:1b", "Gemma3 1B", 1024, "Google, 140+ languages, strong multilingual", "text"),
    ("qwen2.5:1.5b", "Qwen2.5 1.5B", 1500, "29 languages, excellent instruction following", "text"),
    ("qwen2.5:3b", "Qwen2.5 3B", 2500, "29 languages, best quality under 3GB", "text"),
    ("qwen3:4b", "Qwen3 4B", 3000, "100+ languages, best quality for 8GB systems", "text"),
    # Vision models (OpenVINO) - extract metadata directly from images
    ("turingevo/Qwen2.5-VL-3B-Instruct-openvino-int4", "Qwen2.5-VL 3B (Vision)", 2000, "Vision model - extracts metadata from document images", "vision"),
]


# Legacy: Keep VISION_LLM_MODELS for backward compatibility
VISION_LLM_MODELS = [
    (m[0], m[1], m[2], m[3]) for m in RECOMMENDED_MODELS if len(m) > 4 and m[4] == "vision"
]


class LLMConfig(BaseModel):
    """Configuration for LLM prompts and model."""
    model: str = ""  # Ollama model name (empty = use default from OLLAMA_MODEL env)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_de: str = DEFAULT_USER_PROMPT_DE
    user_prompt_en: str = DEFAULT_USER_PROMPT_EN


# Default Vision LLM prompts - for direct image analysis
# Note: {num_pages} placeholder is replaced with actual page count
DEFAULT_VISION_PROMPT_DE = """Analysiere dieses Dokument und extrahiere die Informationen.

WICHTIG für das "topic" Feld:
1. Suche nach der HAUPTÜBERSCHRIFT oder dem BETREFF des Dokuments
2. Bei offiziellen Briefen: Suche nach "Betreff:", "Betrifft:", oder einer fett gedruckten Überschrift
3. Bei Formularen: Nutze den Titel des Formulars (z.B. "Vorladung", "Bescheid", "Mitteilung")
4. Der topic sollte den GRUND oder ZWECK des Dokuments beschreiben, nicht nur die Dokumentart
5. Beispiele: "Vorladung zur Vernehmung", "Kündigung Arbeitsvertrag", "Mahnung Rechnung 12345"

WICHTIG für "identifiers" (Kennzeichen/Referenznummern):
1. Suche nach ALLEN Referenznummern, Aktenzeichen, Vorgangsnummern, Kundennummern, Rechnungsnummern
2. Beispiele: Vorgangsnummer, Aktenzeichen, Rechnungsnummer, Kundennummer, Auftragsnummer
3. Jedes Kennzeichen hat einen "type" (was es ist) und einen "value" (der Wert)

Lies das Dokument sorgfältig und extrahiere die TATSÄCHLICHEN Werte aus dem Dokumentinhalt.
Kopiere NICHT die Beispiele - extrahiere die echten Daten aus dem Bild.

Antworte NUR mit einem JSON-Objekt:
{
    "sender": "<extrahiere den tatsächlichen Absender aus dem Dokument>",
    "receiver": "<extrahiere den tatsächlichen Empfänger oder leer lassen>",
    "document_date": "<YYYY-MM-DD Format oder leer>",
    "topic": "<extrahiere den tatsächlichen Betreff/Titel des Dokuments>",
    "summary": "<schreibe eine kurze Zusammenfassung des Dokumentinhalts>",
    "document_type": "<wähle: letter|invoice|payslip|contract|receipt|statement|notification|summons|certificate|form|report|other>",
    "identifiers": [
        {"type": "<Art der Nummer z.B. Rechnungsnummer>", "value": "<der tatsächliche Wert>"}
    ],
    "iban": "<extrahiere IBAN falls vorhanden oder leer>",
    "bic": "<extrahiere BIC falls vorhanden oder leer>",
    "due_date": "<YYYY-MM-DD falls Fälligkeitsdatum vorhanden oder leer>"
}

WICHTIG: Ersetze alle <...> Platzhalter mit den echten Werten aus dem Dokument!"""

DEFAULT_VISION_PROMPT_EN = """Analyze this document and extract the information.

IMPORTANT for the "topic" field:
1. Look for the MAIN HEADING or SUBJECT of the document
2. For official letters: Look for "Subject:", "Re:", or a bold heading
3. For forms: Use the form title (e.g. "Summons", "Notice", "Certificate")
4. The topic should describe the PURPOSE or REASON of the document, not just the document type
5. Examples: "Summons for Questioning", "Employment Contract Termination", "Payment Reminder Invoice 12345"

IMPORTANT for "identifiers" (reference numbers):
1. Look for ALL reference numbers, case numbers, transaction numbers, customer numbers, invoice numbers
2. Examples: Reference Number, Case Number, Invoice Number, Customer Number, Order Number
3. Each identifier has a "type" (what it is) and a "value" (the actual number/code)

Read the document carefully and extract the ACTUAL values from the document content.
Do NOT copy the examples - extract the real data from the image.

Respond ONLY with a JSON object:
{
    "sender": "<extract the actual sender from the document>",
    "receiver": "<extract the actual recipient or leave empty>",
    "document_date": "<YYYY-MM-DD format or empty>",
    "topic": "<extract the actual subject/title of the document>",
    "summary": "<write a brief summary of the document content>",
    "document_type": "<choose: letter|invoice|payslip|contract|receipt|statement|notification|summons|certificate|form|report|other>",
    "identifiers": [
        {"type": "<type of number e.g. Invoice Number>", "value": "<the actual value>"}
    ],
    "iban": "<extract IBAN if present or empty>",
    "bic": "<extract BIC if present or empty>",
    "due_date": "<YYYY-MM-DD if due date present or empty>"
}

IMPORTANT: Replace all <...> placeholders with real values from the document!"""


class VisionLLMConfig(BaseModel):
    """Configuration for Vision LLM (Qwen2.5-VL with OpenVINO)."""
    enabled: bool = False  # Whether to use vision LLM for metadata extraction
    model_path: str = "models/Qwen3-VL-2B-Instruct-int4"  # Local path to model
    device: str = "CPU"  # Inference device: CPU, GPU, or NPU
    # Image resizing for performance optimization
    # Qwen2.5-VL uses 14x14 patches, dimensions should be multiples of 28
    # Recommended: 256*28*28 (~200k) for speed, 1280*28*28 (~1M) for quality
    max_pixels: int = 512 * 28 * 28  # ~401k pixels - balanced for iGPU INT4
    min_pixels: int = 4 * 28 * 28  # ~3.1k pixels - minimum size
    # Vision LLM prompts (separate from text LLM prompts)
    prompt_de: str = DEFAULT_VISION_PROMPT_DE
    prompt_en: str = DEFAULT_VISION_PROMPT_EN


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    concurrent_workers: int = 1  # Number of documents to process simultaneously
    batch_size: int = 10  # Max documents to fetch per processing cycle


class AppConfig(BaseModel):
    """Main application configuration."""
    incoming_share: IncomingShareConfig = Field(default_factory=IncomingShareConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vision_llm: VisionLLMConfig = Field(default_factory=VisionLLMConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    def save(self) -> None:
        """Save configuration to disk with encrypted passwords."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create a copy for saving with encrypted passwords
        data = self.model_dump()
        
        # Encrypt passwords before saving
        if data["incoming_share"]["credentials"]["password"]:
            data["incoming_share"]["credentials"]["password"] = encrypt_value(
                data["incoming_share"]["credentials"]["password"]
            )
        if data["storage"]["credentials"]["password"]:
            data["storage"]["credentials"]["password"] = encrypt_value(
                data["storage"]["credentials"]["password"]
            )
        
        CONFIG_FILE.write_text(json.dumps(data, indent=2))
        CONFIG_FILE.chmod(0o600)
    
    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from disk and decrypt passwords."""
        if not CONFIG_FILE.exists():
            return cls()
        
        try:
            data = json.loads(CONFIG_FILE.read_text())
            
            # Decrypt passwords after loading
            if data.get("incoming_share", {}).get("credentials", {}).get("password"):
                data["incoming_share"]["credentials"]["password"] = decrypt_value(
                    data["incoming_share"]["credentials"]["password"]
                )
            if data.get("storage", {}).get("credentials", {}).get("password"):
                data["storage"]["credentials"]["password"] = decrypt_value(
                    data["storage"]["credentials"]["password"]
                )
            
            return cls(**data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()
