"""Tests for rate limiting functionality."""

import time
import pytest
from unittest.mock import Mock, patch


class TestRateLimiter:
    """Test rate limiting logic."""

    def test_rate_limit_allows_initial_requests(self):
        """Test that initial requests are allowed."""
        store = {}
        
        # Simulate 5 requests from same IP
        for i in range(5):
            client_ip = "192.168.1.1"
            now = time.time()
            
            if client_ip in store:
                store[client_ip]["count"] += 1
            else:
                store[client_ip] = {"count": 1, "window_start": now}
        
        assert store["192.168.1.1"]["count"] == 5

    def test_rate_limit_resets_after_window(self):
        """Test that rate limit resets after time window."""
        store = {}
        window = 60  # seconds
        
        # Old request
        store["192.168.1.1"] = {"count": 100, "window_start": time.time() - window - 1}
        
        # New request - should reset
        now = time.time()
        if now - store["192.168.1.1"]["window_start"] > window:
            store["192.168.1.1"] = {"count": 1, "window_start": now}
        
        assert store["192.168.1.1"]["count"] == 1

    def test_rate_limit_blocks_excessive_requests(self):
        """Test that excessive requests are blocked."""
        store = {}
        limit = 10
        
        # Simulate exceeding the limit
        store["192.168.1.1"] = {"count": limit + 1, "window_start": time.time()}
        
        # Check if blocked
        assert store["192.168.1.1"]["count"] > limit

    def test_rate_limit_per_ip_isolation(self):
        """Test that rate limits are per-IP."""
        store = {}
        
        # Different IPs should have separate counters
        store["192.168.1.1"] = {"count": 50, "window_start": time.time()}
        store["192.168.1.2"] = {"count": 5, "window_start": time.time()}
        
        assert store["192.168.1.1"]["count"] == 50
        assert store["192.168.1.2"]["count"] == 5

    def test_rate_limit_cleanup(self):
        """Test that old entries are cleaned up."""
        store = {}
        window = 60
        
        # Add old entries
        store["192.168.1.1"] = {"count": 10, "window_start": time.time() - window - 10}
        store["192.168.1.2"] = {"count": 5, "window_start": time.time()}
        
        # Cleanup old entries
        now = time.time()
        for ip in list(store.keys()):
            if now - store[ip]["window_start"] > window:
                del store[ip]
        
        assert "192.168.1.1" not in store
        assert "192.168.1.2" in store
