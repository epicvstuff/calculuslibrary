"""Type stubs for the calculus library."""

from typing import Callable, List, Tuple, Union, Literal
from enum import Enum

__version__: str
__author__: str

# Type aliases
ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[List[float]], float]
VectorValuedFunction = Callable[[List[float]], List[float]]

# Differentiation

def derivative(
    f: ScalarFunction,
    x: float,
    n: int = 1,
    h: float = 1e-6
) -> float:
    """Compute the nth derivative of f at point x."""
    ...

def first_derivative(
    f: ScalarFunction,
    x: float,
    h: float = 1e-6
) -> float:
    """Compute first derivative using central difference method."""
    ...

def second_derivative(
    f: ScalarFunction,
    x: float,
    h: float = 1e-4
) -> float:
    """Compute second derivative using central difference method."""
    ...

def nth_derivative(
    f: ScalarFunction,
    x: float,
    n: int,
    h: float = 1e-3
) -> float:
    """Compute nth derivative using recursive central differences."""
    ...

def forward_difference(
    f: ScalarFunction,
    x: float,
    h: float = 1e-6
) -> float:
    """Compute derivative using forward difference (less accurate)."""
    ...

def backward_difference(
    f: ScalarFunction,
    x: float,
    h: float = 1e-6
) -> float:
    """Compute derivative using backward difference (less accurate)."""
    ...

def derivative_richardson(
    f: ScalarFunction,
    x: float,
    h: float = 1e-4
) -> float:
    """Compute derivative using Richardson extrapolation (higher accuracy)."""
    ...

# Integration

def integrate(
    f: ScalarFunction,
    a: float,
    b: float,
    method: Literal['simpson', 'trapezoidal', 'simpson38', 'gauss', 'romberg'] = 'simpson',
    n: int = 1000
) -> float:
    """Integrate f(x) from a to b using the specified method."""
    ...

def quad(
    f: ScalarFunction,
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 50
) -> float:
    """Adaptive quadrature integration with automatic error control."""
    ...

def trapezoidal(
    f: ScalarFunction,
    a: float,
    b: float,
    n: int = 1000
) -> float:
    """Integrate using the trapezoidal rule."""
    ...

def simpson(
    f: ScalarFunction,
    a: float,
    b: float,
    n: int = 1000
) -> float:
    """Integrate using Simpson's rule."""
    ...

def romberg(
    f: ScalarFunction,
    a: float,
    b: float,
    max_iter: int = 20,
    tol: float = 1e-10
) -> float:
    """Integrate using Romberg's method (Richardson extrapolation)."""
    ...

def gauss_legendre(
    f: ScalarFunction,
    a: float,
    b: float,
    n: int = 100
) -> float:
    """Integrate using Gauss-Legendre quadrature."""
    ...

# Limits

class Direction(Enum):
    """Direction of limit approach."""
    Both = ...
    Left = ...
    Right = ...

class LimitResult:
    """Result of a limit computation."""
    value: float
    exists: bool
    is_finite: bool
    confidence: float

def limit(
    f: ScalarFunction,
    x: float,
    direction: Literal['both', 'left', 'right', '-', '+'] = 'both',
    tol: float = 1e-10
) -> LimitResult:
    """Compute the limit of f(x) as x approaches a given value."""
    ...

def limit_inf(
    f: ScalarFunction,
    direction: Literal['positive', 'negative', '+', '-'] = 'positive',
    tol: float = 1e-8
) -> LimitResult:
    """Compute the limit of f(x) as x approaches infinity."""
    ...

def limit_value(
    f: ScalarFunction,
    x: float,
    direction: Literal['both', 'left', 'right', '-', '+'] = 'both',
    tol: float = 1e-10
) -> float:
    """Compute limit and return just the value (raises exception if limit doesn't exist)."""
    ...

# Series

def taylor(
    f: ScalarFunction,
    a: float,
    n: int = 10,
    h: float = 1e-3
) -> List[float]:
    """Compute Taylor series coefficients around point a."""
    ...

def maclaurin(
    f: ScalarFunction,
    n: int = 10,
    h: float = 1e-3
) -> List[float]:
    """Compute Maclaurin series coefficients (Taylor series around x=0)."""
    ...

def taylor_eval(
    f: ScalarFunction,
    a: float,
    x: float,
    n: int = 10,
    h: float = 1e-3
) -> float:
    """Evaluate Taylor polynomial approximation at point x."""
    ...

def taylor_eval_with_coeffs(
    coefficients: List[float],
    a: float,
    x: float
) -> float:
    """Evaluate Taylor polynomial using pre-computed coefficients."""
    ...

def taylor_error(
    f: ScalarFunction,
    a: float,
    x: float,
    n: int = 10,
    h: float = 1e-3
) -> float:
    """Estimate the error in Taylor approximation using Lagrange remainder."""
    ...

def radius_of_convergence(coefficients: List[float]) -> float:
    """Estimate the radius of convergence for a Taylor series from its coefficients."""
    ...

class PowerSeries:
    """A power series representation."""
    coefficients: List[float]
    center: float

    def __init__(self, coefficients: List[float], center: float) -> None: ...
    def __call__(self, x: float) -> float: ...

def taylor_series(
    f: ScalarFunction,
    a: float,
    n: int = 10,
    h: float = 1e-3
) -> PowerSeries:
    """Create a PowerSeries object representing the Taylor expansion of f around a."""
    ...

# Multivariable (Python-level functions)

def gradient(
    f: VectorFunction,
    point: List[float],
    h: float = 1e-6
) -> List[float]:
    """Compute the gradient of a multivariable function at a point."""
    ...

def jacobian(
    f: VectorValuedFunction,
    point: List[float],
    h: float = 1e-6
) -> List[List[float]]:
    """Compute the Jacobian matrix of a vector-valued function."""
    ...

def hessian(
    f: VectorFunction,
    point: List[float],
    h: float = 1e-4
) -> List[List[float]]:
    """Compute the Hessian matrix of a scalar function."""
    ...
