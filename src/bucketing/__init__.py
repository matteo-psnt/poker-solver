"""
Bucketing module for poker hand abstraction.

This module provides strategies for grouping similar poker situations
into buckets for tractable CFR training.
"""

from src.bucketing.base import BucketingStrategy

__all__ = ["BucketingStrategy"]
