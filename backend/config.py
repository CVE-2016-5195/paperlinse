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
# Format: (ollama_id, display_name, min_memory_mb, description)
RECOMMENDED_MODELS = [
    ("granite4:350m", "Granite4 350M", 512, "IBM, 12 languages, smallest & fastest"),
    ("qwen2.5:0.5b", "Qwen2.5 0.5B", 600, "29 languages, good JSON output"),
    ("qwen3:0.6b", "Qwen3 0.6B", 768, "100+ languages, good for low-power devices"),
    ("gemma3:1b", "Gemma3 1B", 1024, "Google, 140+ languages, strong multilingual"),
    ("qwen2.5:1.5b", "Qwen2.5 1.5B", 1500, "29 languages, excellent instruction following"),
    ("qwen2.5:3b", "Qwen2.5 3B", 2500, "29 languages, best quality under 3GB"),
    ("qwen3:4b", "Qwen3 4B", 3000, "100+ languages, best quality for 8GB systems"),
]


class LLMConfig(BaseModel):
    """Configuration for LLM document analysis."""
    model: str = ""  # Empty means use default (qwen3:0.6b)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_de: str = DEFAULT_USER_PROMPT_DE
    user_prompt_en: str = DEFAULT_USER_PROMPT_EN


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    concurrent_workers: int = 1  # Number of documents to process simultaneously
    batch_size: int = 10  # Max documents to fetch per processing cycle


class AppConfig(BaseModel):
    """Main application configuration."""
    incoming_share: IncomingShareConfig = Field(default_factory=IncomingShareConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
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
