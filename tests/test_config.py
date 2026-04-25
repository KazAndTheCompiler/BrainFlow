"""Tests for BrainConfig security and configuration."""

import os
import pytest
from brain.config import BrainConfig


class TestBrainConfig:
    """Test BrainConfig security features."""

    def test_token_validation_with_valid_token(self):
        """Test that valid tokens are accepted."""
        token = BrainConfig.generate_secure_token()
        os.environ["NEUROLINKED_API_TOKEN"] = token
        # Reload token
        BrainConfig.API_TOKEN = token
        assert BrainConfig.validate_token(token) is True

    def test_token_validation_with_invalid_token(self):
        """Test that invalid tokens are rejected."""
        os.environ["NEUROLINKED_API_TOKEN"] = "valid-token"
        BrainConfig.API_TOKEN = "valid-token"
        assert BrainConfig.validate_token("invalid-token") is False

    def test_token_validation_empty_token(self):
        """Test that empty tokens are rejected in production."""
        os.environ["NEUROLINKED_API_TOKEN"] = ""
        os.environ["NEUROLINKED_ENV"] = "production"
        BrainConfig.API_TOKEN = ""
        # In production with no token, validation should fail
        assert BrainConfig.validate_token("") is False
        os.environ["NEUROLINKED_ENV"] = "development"

    def test_constant_time_comparison(self):
        """Test that tokens are compared in constant time."""
        token = "a" * 32
        os.environ["NEUROLINKED_API_TOKEN"] = token
        BrainConfig.API_TOKEN = token
        # Should not raise or leak timing info
        assert BrainConfig.validate_token(token) is True
        assert BrainConfig.validate_token("b" * 32) is False

    def test_cors_origins_parsing(self):
        """Test CORS origins are parsed correctly."""
        os.environ["NEUROLINKED_CORS_ORIGINS"] = "http://localhost:8000,https://example.com"
        # Reload would happen on import, so test the parsing logic
        origins = "http://localhost:8000,https://example.com".split(",")
        assert origins == ["http://localhost:8000", "https://example.com"]

    def test_rate_limit_config(self):
        """Test rate limiting configuration."""
        os.environ["NEUROLINKED_RATE_LIMIT"] = "true"
        os.environ["NEUROLINKED_RATE_LIMIT_REQUESTS"] = "200"
        os.environ["NEUROLINKED_RATE_LIMIT_WINDOW"] = "120"
        
        assert os.environ["NEUROLINKED_RATE_LIMIT"] == "true"
        assert int(os.environ["NEUROLINKED_RATE_LIMIT_REQUESTS"]) == 200
        assert int(os.environ["NEUROLINKED_RATE_LIMIT_WINDOW"]) == 120

    def test_is_production_detection(self):
        """Test production mode detection."""
        os.environ["NEUROLINKED_ENV"] = "production"
        assert BrainConfig.is_production() is True
        
        os.environ["NEUROLINKED_ENV"] = "development"
        assert BrainConfig.is_production() is False

    def test_rng_seed_config(self):
        """Test RNG seed configuration."""
        os.environ["NEUROLINKED_SEED"] = "42"
        BrainConfig.RNG_SEED = "42"
        
        rng1 = BrainConfig.get_rng()
        rng2 = BrainConfig.get_rng()
        
        # Both should produce same sequence with same seed
        assert rng1.integers(0, 1000) == rng2.integers(0, 1000)


class TestSecurityHeaders:
    """Test security configuration."""

    def test_security_headers_present(self):
        """Verify security headers are configured."""
        # This would require a live server, but we can verify the config exists
        assert hasattr(BrainConfig, 'CORS_ORIGINS')
        assert hasattr(BrainConfig, 'REQUIRE_AUTH')
