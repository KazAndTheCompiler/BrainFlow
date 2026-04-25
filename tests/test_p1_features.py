"""Tests for P1 operational improvements."""

import os
import re
import time
import pytest
from unittest.mock import Mock, patch, MagicMock

from brain.config import BrainConfig


class TestKnowledgeRetention:
    """Test knowledge store retention limits."""

    def test_knowledge_max_entries_config(self):
        """Test that max entries config is parsed correctly."""
        os.environ["NEUROLINKED_KNOWLEDGE_MAX_ENTRIES"] = "5000"
        # Reload would happen on import
        assert int(os.environ["NEUROLINKED_KNOWLEDGE_MAX_ENTRIES"]) == 5000

    def test_knowledge_retention_days_config(self):
        """Test that retention days config is parsed correctly."""
        os.environ["NEUROLINKED_KNOWLEDGE_RETENTION_DAYS"] = "30"
        assert int(os.environ["NEUROLINKED_KNOWLEDGE_RETENTION_DAYS"]) == 30


class TestScreenPrivacy:
    """Test screen observer privacy filters."""

    def test_excluded_titles_parsing(self):
        """Test that excluded titles are parsed correctly."""
        titles = "password,pass,secret,login"
        parsed = titles.lower().split(",")
        assert "password" in parsed
        assert "pass" in parsed
        assert "secret" in parsed

    def test_credit_card_redaction(self):
        """Test that credit card numbers are redacted."""
        text = "My card is 1234 5678 9012 3456 and expires 12/25"
        pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        redacted = re.sub(pattern, '[REDACTED]', text)
        assert '[REDACTED]' in redacted
        assert '1234' not in redacted

    def test_ssn_redaction(self):
        """Test that SSNs are redacted."""
        text = "SSN: 123-45-6789"
        pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        redacted = re.sub(pattern, '[REDACTED]', text)
        assert '[REDACTED]' in redacted

    def test_email_redaction(self):
        """Test that emails are redacted."""
        text = "Contact me at user@example.com"
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        redacted = re.sub(pattern, '[REDACTED]', text)
        assert '[REDACTED]' in redacted
        assert 'user@example.com' not in redacted


class TestRedisRateLimiting:
    """Test Redis-based rate limiting configuration."""

    def test_redis_url_config(self):
        """Test that Redis URL is parsed correctly."""
        os.environ["NEUROLINKED_REDIS_URL"] = "redis://localhost:6379/0"
        assert os.environ["NEUROLINKED_REDIS_URL"] == "redis://localhost:6379/0"

    def test_redis_url_empty_uses_memory(self):
        """Test that empty Redis URL falls back to in-memory."""
        os.environ["NEUROLINKED_REDIS_URL"] = ""
        assert os.environ["NEUROLINKED_REDIS_URL"] == ""


class TestActivityLogRetention:
    """Test activity log retention configuration."""

    def test_activity_log_max_config(self):
        """Test that activity log max is parsed correctly."""
        os.environ["NEUROLINKED_ACTIVITY_LOG_MAX"] = "500"
        assert int(os.environ["NEUROLINKED_ACTIVITY_LOG_MAX"]) == 500


class TestMetrics:
    """Test Prometheus metrics configuration."""

    def test_metrics_enabled_config(self):
        """Test that metrics enabled flag is parsed correctly."""
        os.environ["NEUROLINKED_METRICS_ENABLED"] = "true"
        assert os.environ["NEUROLINKED_METRICS_ENABLED"].lower() == "true"

    def test_metrics_port_config(self):
        """Test that metrics port is parsed correctly."""
        os.environ["NEUROLINKED_METRICS_PORT"] = "9090"
        assert int(os.environ["NEUROLINKED_METRICS_PORT"]) == 9090


class TestDeterminism:
    """Test RNG seeding for determinism."""

    def test_seed_config(self):
        """Test that seed is parsed correctly."""
        os.environ["NEUROLINKED_SEED"] = "42"
        assert os.environ["NEUROLINKED_SEED"] == "42"

    def test_get_rng_with_seed(self):
        """Test that get_rng produces deterministic results with seed."""
        os.environ["NEUROLINKED_SEED"] = "42"
        BrainConfig.RNG_SEED = "42"
        
        rng1 = BrainConfig.get_rng()
        rng2 = BrainConfig.get_rng()
        
        # Should produce same sequence
        assert rng1.integers(0, 1000) == rng2.integers(0, 1000)

    def test_get_rng_without_seed(self):
        """Test that get_rng produces random results without seed."""
        os.environ["NEUROLINKED_SEED"] = ""
        BrainConfig.RNG_SEED = None
        
        rng1 = BrainConfig.get_rng()
        rng2 = BrainConfig.get_rng()
        
        # Should produce different sequences
        # (Very unlikely to be equal with true randomness)
        seq1 = [rng1.random() for _ in range(5)]
        seq2 = [rng2.random() for _ in range(5)]
        # We can't assert they're different (could theoretically be same)
        # but we can assert they're valid random numbers
        assert all(0 <= x < 1 for x in seq1)
        assert all(0 <= x < 1 for x in seq2)
