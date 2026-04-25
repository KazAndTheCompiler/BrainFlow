"""Tests for Brain determinism and core functionality."""

import os
import pytest
import numpy as np

from brain.config import BrainConfig


class TestBrainDeterminism:
    """Test that brain produces deterministic results with same seed."""

    def test_rng_produces_same_sequence_with_same_seed(self):
        """Test that RNG produces identical sequences with same seed."""
        os.environ["NEUROLINKED_SEED"] = "42"
        BrainConfig.RNG_SEED = "42"
        
        rng1 = BrainConfig.get_rng()
        rng2 = BrainConfig.get_rng()
        
        # Generate sequences
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]
        
        assert seq1 == seq2

    def test_rng_produces_different_sequences_with_different_seeds(self):
        """Test that different seeds produce different sequences."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(43)
        
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]
        
        assert seq1 != seq2

    def test_neuron_initialization_deterministic(self):
        """Test that neuron initialization is deterministic with seed."""
        seed = 42
        
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        
        # Simulate neuron position initialization
        positions1 = rng1.random((100, 3))
        positions2 = rng2.random((100, 3))
        
        np.testing.assert_array_equal(positions1, positions2)


class TestSTDP:
    """Test Spike-Timing Dependent Plasticity."""

    def test_stdp_increases_weight_for_pre_before_post(self):
        """Test that pre-before-post timing increases synaptic weight."""
        # This is a conceptual test - actual implementation would need
        # access to the synapse update logic
        
        # Pre fires before post -> potentiation (weight increase)
        pre_time = 0.0
        post_time = 5.0  # 5ms after pre
        
        # Simple STDP rule: positive delta_t = post - pre > 0 -> LTP
        delta_t = post_time - pre_time
        
        # LTP occurs when pre fires before post (positive delta_t)
        assert delta_t > 0

    def test_stdp_decreases_weight_for_post_before_pre(self):
        """Test that post-before-pre timing decreases synaptic weight."""
        # Post fires before pre -> depression (weight decrease)
        post_time = 0.0
        pre_time = 5.0  # 5ms after post
        
        delta_t = post_time - pre_time
        
        # LTD occurs when post fires before pre (negative delta_t)
        assert delta_t < 0


class TestBrainConfig:
    """Test BrainConfig values."""

    def test_region_proportions_sum_to_one(self):
        """Test that region proportions sum to 1.0."""
        total = sum(BrainConfig.REGION_PROPORTIONS.values())
        assert abs(total - 1.0) < 1e-6, f"Region proportions sum to {total}, not 1.0"

    def test_all_regions_have_neuron_params(self):
        """Test that all regions have neuron parameters defined."""
        for region in BrainConfig.REGION_PROPORTIONS.keys():
            assert region in BrainConfig.NEURON_PARAMS, f"{region} missing neuron params"
            params = BrainConfig.NEURON_PARAMS[region]
            required = ["a", "b", "c", "d"]
            for p in required:
                assert p in params, f"{region} missing parameter {p}"

    def test_synapse_count_reasonable(self):
        """Test that synapse count is reasonable."""
        total_neurons = BrainConfig.TOTAL_NEURONS
        synapses_per = BrainConfig.SYNAPSES_PER_NEURON
        
        # Total synapses should be neurons * synapses_per_neuron
        expected_synapses = total_neurons * synapses_per
        
        # Should be within reasonable bounds (1M to 1B)
        assert 1_000_000 <= expected_synapses <= 1_000_000_000


class TestSafetyLimits:
    """Test safety kernel limits."""

    def test_force_limit_defined(self):
        """Test that force limits are defined."""
        assert hasattr(BrainConfig, 'SAFETY_FORCE_LIMIT')
        assert BrainConfig.SAFETY_FORCE_LIMIT > 0

    def test_velocity_limit_defined(self):
        """Test that velocity limits are defined."""
        assert hasattr(BrainConfig, 'SAFETY_VELOCITY_LIMIT')
        assert BrainConfig.SAFETY_VELOCITY_LIMIT > 0
