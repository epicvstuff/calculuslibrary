"""
Benchmarks comparing calculus library performance against pure Python and scipy.

Run with: pytest benchmarks/ -v --benchmark-only
"""

import math
import pytest

import calculus
from pure_python import (
    derivative_pure,
    second_derivative_pure,
    trapezoidal_pure,
    simpson_pure,
    taylor_coefficients_pure,
)

# Try to import scipy for comparison
try:
    import scipy.integrate
    from scipy.misc import derivative as scipy_derivative
    HAS_SCIPY = True
    HAS_SCIPY_DERIVATIVE = True
except ImportError:
    HAS_SCIPY = False
    HAS_SCIPY_DERIVATIVE = False
except AttributeError:
    # scipy.misc.derivative removed in scipy 1.12+
    import scipy.integrate
    HAS_SCIPY = True
    HAS_SCIPY_DERIVATIVE = False


# Test functions
def f_poly(x):
    return x**3 - 2*x**2 + x - 1

def f_trig(x):
    return math.sin(x) * math.cos(x)

def f_exp(x):
    return math.exp(-x**2)


class TestDerivativeBenchmarks:
    """Benchmarks for differentiation functions."""

    def test_derivative_calculus(self, benchmark):
        """Benchmark: calculus library first derivative."""
        result = benchmark(calculus.derivative, f_poly, 2.0)
        assert abs(result - 5.0) < 1e-5  # 3x² - 4x + 1 at x=2 = 12 - 8 + 1 = 5

    def test_derivative_pure_python(self, benchmark):
        """Benchmark: pure Python first derivative."""
        result = benchmark(derivative_pure, f_poly, 2.0)
        assert abs(result - 5.0) < 1e-5

    @pytest.mark.skipif(not HAS_SCIPY_DERIVATIVE, reason="scipy.misc.derivative not available")
    def test_derivative_scipy(self, benchmark):
        """Benchmark: scipy derivative."""
        result = benchmark(scipy_derivative, f_poly, 2.0, dx=1e-6)
        assert abs(result - 5.0) < 1e-5

    def test_second_derivative_calculus(self, benchmark):
        """Benchmark: calculus library second derivative."""
        result = benchmark(calculus.derivative, f_poly, 2.0, n=2)
        assert abs(result - 8.0) < 0.1  # 6x - 4 at x=2 = 8

    def test_second_derivative_pure_python(self, benchmark):
        """Benchmark: pure Python second derivative."""
        result = benchmark(second_derivative_pure, f_poly, 2.0)
        assert abs(result - 8.0) < 0.1


class TestIntegrationBenchmarks:
    """Benchmarks for integration functions."""

    def test_integrate_simpson_calculus(self, benchmark):
        """Benchmark: calculus library Simpson integration."""
        result = benchmark(calculus.integrate, f_exp, 0.0, 2.0, method='simpson')
        assert abs(result - 0.8820813907624215) < 1e-4

    def test_integrate_simpson_pure_python(self, benchmark):
        """Benchmark: pure Python Simpson integration."""
        result = benchmark(simpson_pure, f_exp, 0.0, 2.0)
        assert abs(result - 0.8820813907624215) < 1e-4

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
    def test_integrate_scipy(self, benchmark):
        """Benchmark: scipy quad integration."""
        result = benchmark(lambda: scipy.integrate.quad(f_exp, 0.0, 2.0)[0])
        assert abs(result - 0.8820813907624215) < 1e-6

    def test_integrate_trapezoidal_calculus(self, benchmark):
        """Benchmark: calculus library trapezoidal integration."""
        result = benchmark(calculus.trapezoidal, f_exp, 0.0, 2.0)
        assert abs(result - 0.8820813907624215) < 1e-4

    def test_integrate_trapezoidal_pure_python(self, benchmark):
        """Benchmark: pure Python trapezoidal integration."""
        result = benchmark(trapezoidal_pure, f_exp, 0.0, 2.0)
        assert abs(result - 0.8820813907624215) < 1e-4

    def test_integrate_romberg_calculus(self, benchmark):
        """Benchmark: calculus library Romberg integration."""
        result = benchmark(calculus.romberg, f_exp, 0.0, 2.0)
        assert abs(result - 0.8820813907624215) < 1e-8

    def test_integrate_gauss_calculus(self, benchmark):
        """Benchmark: calculus library Gauss-Legendre integration."""
        result = benchmark(calculus.gauss_legendre, f_exp, 0.0, 2.0)
        assert abs(result - 0.8820813907624215) < 1e-6


class TestSeriesBenchmarks:
    """Benchmarks for Taylor series functions."""

    def test_taylor_calculus(self, benchmark):
        """Benchmark: calculus library Taylor coefficients."""
        result = benchmark(calculus.taylor, math.exp, 0.0, n=10)
        assert len(result) == 10
        assert abs(result[0] - 1.0) < 0.01

    def test_taylor_pure_python(self, benchmark):
        """Benchmark: pure Python Taylor coefficients."""
        result = benchmark(taylor_coefficients_pure, math.exp, 0.0, n=10)
        assert len(result) == 10
        assert abs(result[0] - 1.0) < 0.01

    def test_taylor_eval_calculus(self, benchmark):
        """Benchmark: calculus library Taylor evaluation."""
        result = benchmark(calculus.taylor_eval, math.sin, 0.0, 0.5, n=10)
        assert abs(result - math.sin(0.5)) < 1e-4


class TestHighVolumeIntegration:
    """Benchmarks with varying workloads."""

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_simpson_varying_n_calculus(self, benchmark, n):
        """Benchmark: calculus Simpson with varying intervals."""
        result = benchmark(calculus.integrate, f_exp, 0.0, 2.0, method='simpson', n=n)
        assert abs(result - 0.8820813907624215) < 0.01

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_simpson_varying_n_pure_python(self, benchmark, n):
        """Benchmark: pure Python Simpson with varying intervals."""
        result = benchmark(simpson_pure, f_exp, 0.0, 2.0, n=n)
        assert abs(result - 0.8820813907624215) < 0.01
