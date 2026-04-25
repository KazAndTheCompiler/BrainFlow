"""Tests for persistence layer with schema validation."""

import json
import os
import tempfile
import shutil
import pytest
import numpy as np

from brain.persistence import (
    _compute_checksum,
    _validate_meta,
    _sanitize_meta,
    SCHEMA_VERSION,
    REQUIRED_META_FIELDS,
)


class TestChecksum:
    """Test checksum computation and verification."""

    def test_checksum_computation(self):
        """Test that checksums are computed consistently."""
        data = {
            "step_count": 1000,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {"dopamine": 0.5, "serotonin": 0.3},
            "saved_at": 1234567890.0,
        }
        
        checksum1 = _compute_checksum(data)
        checksum2 = _compute_checksum(data)
        
        # Same data should produce same checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 16  # SHA256 truncated to 16 chars

    def test_checksum_different_data(self):
        """Test that different data produces different checksums."""
        data1 = {"step_count": 1000, "total_neurons": 10000}
        data2 = {"step_count": 1001, "total_neurons": 10000}
        
        checksum1 = _compute_checksum(data1)
        checksum2 = _compute_checksum(data2)
        
        assert checksum1 != checksum2

    def test_checksum_ignores_checksum_field(self):
        """Test that checksum field is excluded from computation."""
        data = {"step_count": 1000, "checksum": "abc123"}
        
        checksum = _compute_checksum(data)
        
        # Should not include the existing checksum in computation
        data_with_checksum = {**data, "checksum": checksum}
        recomputed = _compute_checksum(data_with_checksum)
        assert checksum == recomputed


class TestSchemaValidation:
    """Test schema validation for meta.json."""

    def test_valid_meta_passes(self):
        """Test that valid meta passes validation."""
        meta = {
            "step_count": 1000,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                "serotonin": 0.3,
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
        }
        
        is_valid, error = _validate_meta(meta)
        assert is_valid is True
        assert error == ""

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        meta = {
            "step_count": 1000,
            # missing total_neurons
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                "serotonin": 0.3,
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
        }
        
        is_valid, error = _validate_meta(meta)
        assert is_valid is False
        assert "total_neurons" in error

    def test_invalid_type_fails(self):
        """Test that invalid types fail validation."""
        meta = {
            "step_count": "not_a_number",  # should be int
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                "serotonin": 0.3,
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
        }
        
        is_valid, error = _validate_meta(meta)
        assert is_valid is False
        assert "step_count" in error

    def test_missing_neuromodulator_fails(self):
        """Test that missing neuromodulators fail validation."""
        meta = {
            "step_count": 1000,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                # missing serotonin, norepinephrine, acetylcholine
            },
            "saved_at": 1234567890.0,
        }
        
        is_valid, error = _validate_meta(meta)
        assert is_valid is False
        assert "serotonin" in error or "neuromodulator" in error

    def test_checksum_validation(self):
        """Test that checksums are validated."""
        meta = {
            "step_count": 1000,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                "serotonin": 0.3,
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
            "checksum": "invalid_checksum",
        }
        
        is_valid, error = _validate_meta(meta)
        assert is_valid is False
        assert "Checksum mismatch" in error


class TestSanitization:
    """Test data sanitization."""

    def test_negative_step_count_clamped(self):
        """Test that negative step counts are clamped."""
        meta = {
            "step_count": -100,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                "serotonin": 0.3,
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
        }
        
        sanitized = _sanitize_meta(meta)
        assert sanitized["step_count"] == 0

    def test_neuromodulators_clamped_to_range(self):
        """Test that neuromodulators are clamped to [0, 1]."""
        meta = {
            "step_count": 1000,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 1.5,  # > 1
                "serotonin": -0.3,  # < 0
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
        }
        
        sanitized = _sanitize_meta(meta)
        assert sanitized["neuromodulators"]["dopamine"] == 1.0
        assert sanitized["neuromodulators"]["serotonin"] == 0.0

    def test_optional_fields_added(self):
        """Test that optional fields get defaults."""
        meta = {
            "step_count": 1000,
            "total_neurons": 10000,
            "development_stage": "JUVENILE",
            "neuromodulators": {
                "dopamine": 0.5,
                "serotonin": 0.3,
                "norepinephrine": 0.4,
                "acetylcholine": 0.6,
            },
            "saved_at": 1234567890.0,
        }
        
        sanitized = _sanitize_meta(meta)
        assert "schema_version" in sanitized
        assert "uptime" in sanitized
        assert "checksum" in sanitized
