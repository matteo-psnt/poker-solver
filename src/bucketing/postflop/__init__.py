"""
Postflop bucketing with suit isomorphism.

This module provides the foundation for combo-level abstraction under suit isomorphism,
which is required for theoretically sound postflop bucketing in poker solvers.

Key concepts:
- Suits are interchangeable until the board creates distinctions
- Canonical representation assigns suits to labels (0,1,2,3) in order of appearance
- Hands are canonicalized relative to the board's suit mapping
- This eliminates duplicate states and ensures deterministic bucket mapping
"""
