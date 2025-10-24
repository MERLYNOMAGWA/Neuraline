import re

class ContentFilter:
    """Basic safety filter to strip PII and unsafe content."""

    def clean(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[email redacted]", text)
        text = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[phone redacted]", text)
        text = text.replace("password", "[redacted]")
        return text