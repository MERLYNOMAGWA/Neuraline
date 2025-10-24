class ResponseValidator:
    """Ensures model outputs are safe and compliant before storage."""

    def clean(self, text: str) -> str:
        if not text:
            return text
        disallowed = ["suicide", "violence", "hate speech"]
        for word in disallowed:
            if word.lower() in text.lower():
                text = text.replace(word, "[content removed]")
        return text