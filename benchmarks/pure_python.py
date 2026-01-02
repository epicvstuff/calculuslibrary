"""
Pure Python implementations of calculus functions for benchmarking.
These are intentionally simple implementations to compare against the C++ library.
"""

import math


def derivative_pure(f, x, h=1e-6):
    """Central difference derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative_pure(f, x, h=1e-4):
    """Second derivative using central difference."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def trapezoidal_pure(f, a, b, n=1000):
    """Trapezoidal rule integration."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def simpson_pure(f, a, b, n=1000):
    """Simpson's rule integration."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)
    return result * h / 3


def taylor_coefficients_pure(f, a, n=10, h=1e-3):
    """Compute Taylor coefficients using finite differences."""
    coeffs = []
    factorial = 1
    for k in range(n):
        if k > 0:
            factorial *= k
        # Compute kth derivative using central differences
        deriv = nth_derivative_pure(f, a, k, h)
        coeffs.append(deriv / factorial)
    return coeffs


def nth_derivative_pure(f, x, n, h=1e-3):
    """Compute nth derivative using finite differences."""
    if n == 0:
        return f(x)

    adjusted_h = h * (2 ** ((n - 1) / 2))

    def binomial(n, k):
        if k < 0 or k > n:
            return 0
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    result = 0
    h_n = adjusted_h ** n
    for k in range(n + 1):
        coef = binomial(n, k)
        if k % 2 == 1:
            coef = -coef
        point = x + (n / 2 - k) * adjusted_h
        result += coef * f(point)

    return result / h_n
