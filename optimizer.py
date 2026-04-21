"""
Steepest Descent Optimizer
Implements all step size methods from: Napitupulu et al. (2018)
IOP Conf. Ser.: Mater. Sci. Eng. 332 012024
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import math


@dataclass
class IterationData:
    k: int
    x: List[float]
    f: float
    grad_norm: float
    step_size: float
    direction: List[float]


@dataclass
class OptimizationResult:
    success: bool
    x_opt: List[float]
    f_opt: float
    iterations: int
    path: List[List[float]]      # list of [x1, x2] at each iteration
    f_values: List[float]
    grad_norms: List[float]
    step_sizes: List[float]
    time_ms: float
    message: str
    method: str


# ─────────────────────────────────────────────
#  Predefined test functions (analytically)
# ─────────────────────────────────────────────

def rosenbrock(x):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2  — minimum at (1,1)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

def rosenbrock_hess(x):
    h11 = 2 - 400*(x[1] - x[0]**2) + 800*x[0]**2
    h12 = -400*x[0]
    h21 = -400*x[0]
    h22 = 200.0
    return np.array([[h11, h12], [h21, h22]])


def beale(x):
    """Beale function — minimum at (3, 0.5)"""
    t1 = (1.5  - x[0] + x[0]*x[1])**2
    t2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    t3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return t1 + t2 + t3

def beale_grad(x):
    a = 1.5   - x[0] + x[0]*x[1]
    b = 2.25  - x[0] + x[0]*x[1]**2
    c = 2.625 - x[0] + x[0]*x[1]**3
    dx = 2*a*(-1 + x[1]) + 2*b*(-1 + x[1]**2) + 2*c*(-1 + x[1]**3)
    dy = 2*a*(x[0]) + 2*b*(2*x[0]*x[1]) + 2*c*(3*x[0]*x[1]**2)
    return np.array([dx, dy])

def beale_hess(x):
    return _numerical_hessian(beale, x)


def booth(x):
    """Booth function — minimum at (1, 3)"""
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def booth_grad(x):
    dx = 2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5)
    dy = 4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5)
    return np.array([dx, dy])

def booth_hess(x):
    return np.array([[10.0, 8.0], [8.0, 10.0]])


def matyas(x):
    """Matyas function — minimum at (0, 0)"""
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]

def matyas_grad(x):
    dx = 0.52*x[0] - 0.48*x[1]
    dy = 0.52*x[1] - 0.48*x[0]
    return np.array([dx, dy])

def matyas_hess(x):
    return np.array([[0.52, -0.48], [-0.48, 0.52]])


def rastrigin(x):
    """Rastrigin 2D — minimum at (0, 0)"""
    return x[0]**2 + x[1]**2 - np.cos(18*x[0]) - np.cos(18*x[1])

def rastrigin_grad(x):
    dx = 2*x[0] + 18*np.sin(18*x[0])
    dy = 2*x[1] + 18*np.sin(18*x[1])
    return np.array([dx, dy])

def rastrigin_hess(x):
    h11 = 2 + 18**2 * np.cos(18*x[0])
    h22 = 2 + 18**2 * np.cos(18*x[1])
    return np.array([[h11, 0.0], [0.0, h22]])


# ─────────────────────────────────────────────
#  Numerical gradient / Hessian
# ─────────────────────────────────────────────

def _numerical_gradient(f, x, eps=1e-7):
    g = np.zeros_like(x, dtype=float)
    fx = f(x)
    for i in range(len(x)):
        xh = x.copy()
        xh[i] += eps
        g[i] = (f(xh) - fx) / eps
    return g

def _numerical_hessian(f, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i,j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4*eps**2)
    return H


# ─────────────────────────────────────────────
#  Function registry
# ─────────────────────────────────────────────

FUNCTIONS = {
    "rosenbrock": {
        "f": rosenbrock,
        "grad": rosenbrock_grad,
        "hess": rosenbrock_hess,
        "default_x0": [-1.2, 1.0],
        "xlim": [-2, 2],
        "ylim": [-1, 3],
        "name": "Rosenbrock",
        "latex": "f(x,y) = (1-x)^2 + 100(y-x^2)^2",
        "optimum": [1.0, 1.0],
        "f_opt": 0.0,
    },
    "beale": {
        "f": beale,
        "grad": beale_grad,
        "hess": beale_hess,
        "default_x0": [1.0, 0.0],
        "xlim": [-4.5, 4.5],
        "ylim": [-4.5, 4.5],
        "name": "Beale",
        "latex": "f(x,y) = (1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2",
        "optimum": [3.0, 0.5],
        "f_opt": 0.0,
    },
    "booth": {
        "f": booth,
        "grad": booth_grad,
        "hess": booth_hess,
        "default_x0": [-4.0, 4.0],
        "xlim": [-10, 10],
        "ylim": [-10, 10],
        "name": "Booth",
        "latex": "f(x,y) = (x+2y-7)^2 + (2x+y-5)^2",
        "optimum": [1.0, 3.0],
        "f_opt": 0.0,
    },
    "matyas": {
        "f": matyas,
        "grad": matyas_grad,
        "hess": matyas_hess,
        "default_x0": [2.0, 2.0],
        "xlim": [-3, 3],
        "ylim": [-3, 3],
        "name": "Matyas",
        "latex": "f(x,y) = 0.26(x^2+y^2) - 0.48xy",
        "optimum": [0.0, 0.0],
        "f_opt": 0.0,
    },
    "rastrigin": {
        "f": rastrigin,
        "grad": rastrigin_grad,
        "hess": rastrigin_hess,
        "default_x0": [-1.0, 1.0],
        "xlim": [-2, 2],
        "ylim": [-2, 2],
        "name": "Rastrigin 2D",
        "latex": "f(x,y) = x^2+y^2-\\cos(18x)-\\cos(18y)",
        "optimum": [0.0, 0.0],
        "f_opt": -2.0,
    },
}


# ─────────────────────────────────────────────
#  Step size methods
# ─────────────────────────────────────────────

def step_cauchy(f, grad, hess, x, d):
    """
    C step: exact line search via quadratic formula
    α = (gᵀg) / (gᵀHg)
    Falls back to backtracking if Hg term is non-positive.
    """
    g = -d  # g = gradient = -direction
    Hg = hess(x) @ g
    denom = g @ Hg
    if denom > 1e-15:
        return (g @ g) / denom
    # fallback: golden section
    return _golden_section(f, x, d)


def step_armijo(f, grad_fn, x, d, s=1.0, beta=0.75, sigma=0.38):
    """
    A step: Armijo inexact line search.
    Find largest α in {s, sβ, sβ², ...} such that
    f(x + αd) ≤ f(x) + σα·∇f(x)ᵀd
    """
    g = grad_fn(x)
    slope = g @ d  # should be negative (descent)
    if slope >= 0:
        return 1e-4
    alpha = s
    fx = f(x)
    for _ in range(100):
        if f(x + alpha * d) <= fx + sigma * alpha * slope:
            return alpha
        alpha *= beta
    return alpha


def step_backtracking(f, grad_fn, x, d, beta=0.75, sigma=0.0001):
    """
    B step: Backtracking with sufficient decrease (Armijo-like, fixed init α=1).
    """
    g = grad_fn(x)
    slope = g @ d
    if slope >= 0:
        return 1e-4
    alpha = 1.0
    fx = f(x)
    for _ in range(200):
        if f(x + alpha * d) <= fx + sigma * alpha * slope:
            return alpha
        alpha *= beta
    return alpha


def step_bb1(s_prev, y_prev):
    """
    BB1: α = (sᵀy) / (yᵀy)
    where s = x_k - x_{k-1}, y = g_k - g_{k-1}
    """
    yy = y_prev @ y_prev
    if yy < 1e-30:
        return 0.01
    return max(1e-10, min((s_prev @ y_prev) / yy, 1.0))


def step_bb2(s_prev, y_prev):
    """
    BB2: α = (sᵀs) / (sᵀy)
    """
    sy = s_prev @ y_prev
    if abs(sy) < 1e-30:
        return 0.01
    ss = s_prev @ s_prev
    return max(1e-10, min(ss / sy, 1.0))


def step_elimination(f, x, d, g, alpha_prev, f_prev):
    """
    EL step: estimation without Hessian via quadratic fit.
    α_k = (gᵀg · α_{k-1}²) / (2·[f(x+α_{k-1}d) - f(x) + α_{k-1}·gᵀg])
    Falls back to backtracking on degenerate cases.
    """
    if alpha_prev is None or alpha_prev <= 0:
        return step_backtracking(f, lambda xx: -d, x, d)
    
    gg = g @ g
    fx = f(x)
    f_trial = f(x + alpha_prev * d)
    denom = 2 * (f_trial - fx + alpha_prev * gg)
    
    if abs(denom) < 1e-30 or denom <= 0:
        return step_backtracking(f, lambda xx: -d, x, d)
    
    alpha = (gg * alpha_prev**2) / denom
    return max(1e-10, min(alpha, 10.0))


def _golden_section(f, x, d, lo=0.0, hi=2.0, tol=1e-8):
    """1D exact line search via golden section."""
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi * (b - a)
    e = a + phi * (b - a)
    fc, fe = f(x + c*d), f(x + e*d)
    for _ in range(100):
        if abs(b - a) < tol:
            break
        if fc < fe:
            b = e; e = c; fe = fc
            c = b - phi*(b-a); fc = f(x+c*d)
        else:
            a = c; c = e; fc = fe
            e = a + phi*(b-a); fe = f(x+e*d)
    return (a + b) / 2


# ─────────────────────────────────────────────
#  Main optimization loop
# ─────────────────────────────────────────────

def optimize(
    func_name: str,
    method: str,
    x0: List[float],
    max_iter: int = 5000,
    tol: float = 1e-6,
    # Armijo params
    armijo_s: float = 1.0,
    armijo_beta: float = 0.75,
    armijo_sigma: float = 0.38,
    # Backtracking params
    bt_beta: float = 0.75,
    bt_sigma: float = 0.0001,
    # BB initial step
    bb_alpha0: float = 0.01,
    # EL initial step
    el_alpha0: float = 0.1,
) -> OptimizationResult:
    import time

    # ── resolve function ──────────────────────
    info = FUNCTIONS.get(func_name)
    if info is None:
        return OptimizationResult(
            success=False, x_opt=x0, f_opt=float('nan'),
            iterations=0, path=[], f_values=[], grad_norms=[],
            step_sizes=[], time_ms=0, message="Unknown function",
            method=method
        )
    f        = info["f"]
    grad_fn  = info["grad"]
    hess_fn  = info["hess"]

    x = np.array(x0, dtype=float)
    path        = [x.tolist()]
    f_values    = [f(x)]
    grad_norms  = []
    step_sizes  = []

    alpha_prev  = bb_alpha0 if method in ("BB1", "BB2") else el_alpha0
    x_prev      = None
    g_prev      = None
    f_prev      = None

    t0 = time.time()

    for k in range(max_iter):
        g = grad_fn(x)
        gnorm = float(np.linalg.norm(g))
        grad_norms.append(gnorm)

        # ── stopping ─────────────────────────
        if gnorm < tol:
            break

        d = -g  # steepest descent direction

        # ── step size selection ───────────────
        if method == "C":
            alpha = step_cauchy(f, grad_fn, hess_fn, x, d)
        elif method == "A":
            alpha = step_armijo(f, grad_fn, x, d, s=armijo_s,
                                beta=armijo_beta, sigma=armijo_sigma)
        elif method == "B":
            alpha = step_backtracking(f, grad_fn, x, d,
                                      beta=bt_beta, sigma=bt_sigma)
        elif method == "BB1":
            if x_prev is not None and g_prev is not None:
                s_k = x - x_prev
                y_k = g - g_prev
                alpha = step_bb1(s_k, y_k)
            else:
                alpha = bb_alpha0
        elif method == "BB2":
            if x_prev is not None and g_prev is not None:
                s_k = x - x_prev
                y_k = g - g_prev
                alpha = step_bb2(s_k, y_k)
            else:
                alpha = bb_alpha0
        elif method == "EL":
            alpha = step_elimination(f, x, d, g, alpha_prev, f_prev)
        else:
            alpha = 0.01  # fallback

        # ── guard step size ───────────────────
        alpha = max(1e-12, min(alpha, 10.0))

        # ── update ───────────────────────────
        x_prev  = x.copy()
        g_prev  = g.copy()
        f_prev  = f(x)
        x       = x + alpha * d
        alpha_prev = alpha

        path.append(x.tolist())
        f_values.append(float(f(x)))
        step_sizes.append(float(alpha))

        # extra stopping: step too small
        step_norm = float(np.linalg.norm(alpha * d))
        if step_norm < tol:
            break

    elapsed = (time.time() - t0) * 1000  # ms

    # final gradient for last point
    g_final = grad_fn(x)
    grad_norms.append(float(np.linalg.norm(g_final)))

    converged = float(np.linalg.norm(grad_fn(x))) < tol * 100

    return OptimizationResult(
        success=converged,
        x_opt=x.tolist(),
        f_opt=float(f(x)),
        iterations=len(step_sizes),
        path=path,
        f_values=f_values,
        grad_norms=grad_norms,
        step_sizes=step_sizes,
        time_ms=elapsed,
        message="Converged" if converged else f"Max iterations ({max_iter}) reached",
        method=method,
    )
