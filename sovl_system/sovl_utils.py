import torch
import math

"""
Numerical utilities for safe floating-point operations
Place this in your project root or utils/ directory
"""

def float_equal(a: float, b: float, rel_tol: float = 1e-5, abs_tol: float = 1e-8) -> bool:
    """Safe floating point comparison with relative and absolute tolerance"""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def float_gt(a: float, b: float, rel_tol: float = 1e-5) -> bool:
    """Safe greater-than comparison"""
    return a > b and not float_equal(a, b, rel_tol)

def float_lt(a: float, b: float, rel_tol: float = 1e-5) -> bool:
    """Safe less-than comparison"""
    return a < b and not float_equal(a, b, rel_tol)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with protection against zero denominator"""
    return numerator / (denominator + 1e-10) if abs(denominator) > 1e-10 else default
