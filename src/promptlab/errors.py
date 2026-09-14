"""Errors used by the Week 2 local model lab."""


class UnknownModelError(ValueError):
    """Raised when a model identifier is not present in the configured model table."""


class TransientProviderError(Exception):
    """Timeout, connection failure, or temporary Ollama/server failure. Retried."""


class PermanentProviderError(Exception):
    """Non-retryable request failure.

    Raised for a malformed request, unavailable model, unsupported parameter,
    or other non-retryable provider error.
    """


class TruncatedResponseError(Exception):
    """Ollama reported that the output token ceiling was reached. Not retried."""
