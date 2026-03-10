"""
Pydantic v2 models for the basis detection API.

Used by routes_basis.py to serialize responses.
"""
from pydantic import BaseModel
from typing import List


class BasisResult(BaseModel):
    """Metrics for a single transform domain."""
    domain: str
    sparsity: float
    reconstruction_error: float
    num_coefficients: int


class BasisResponse(BaseModel):
    """Response from the basis analysis endpoint."""
    best_basis: str
    results: List[BasisResult]


class BasisRecommendation(BaseModel):
    """Extended response with a human-readable recommendation."""
    best_basis: str
    recommendation: str  # e.g. "Use DWT Symlet-8: highest sparsity (0.87) for this signal."
    results: List[BasisResult]
