 """
FastAPI Backend — Steepest Descent Optimizer
Endpoints:
  GET  /functions          → list available functions + metadata
  POST /optimize           → run one method, return full path
  POST /compare            → run all methods, return comparison
  POST /contour            → compute contour data for a function
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np

from optimizer import optimize, FUNCTIONS, _numerical_gradient

app = FastAPI(
    title="Steepest Descent Optimizer API",
    description="Interactive unconstrained optimization — Napitupulu et al. (2018)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    func_name: str = Field("rosenbrock", description="Function key")
    method: str    = Field("A", description="C|A|B|BB1|BB2|EL")
    x0: List[float] = Field([-1.2, 1.0])
    max_iter: int  = Field(5000, ge=1, le=20000)
    tol: float     = Field(1e-6, gt=0)
    # Armijo
    armijo_s: float     = 1.0
    armijo_beta: float  = 0.75
    armijo_sigma: float = 0.38
    # Backtracking
    bt_beta: float  = 0.75
    bt_sigma: float = 0.0001
    # BB
    bb_alpha0: float = 0.01
    # EL
    el_alpha0: float = 0.1



class CompareRequest(BaseModel):
    func_name: str    = "rosenbrock"
    methods: List[str] = ["C", "A", "B", "BB1", "BB2", "EL"]
    x0: List[float]   = [-1.2, 1.0]
    max_iter: int      = 5000
    tol: float         = 1e-6


class ContourRequest(BaseModel):
    func_name: str    = "rosenbrock"
    xlim: List[float] = [-2.0, 2.0]
    ylim: List[float] = [-1.0, 3.0]
    resolution: int   = Field(80, ge=20, le=300)


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "steepest-descent-api"}


@app.get("/functions")
def list_functions():
    """Return metadata for all built-in functions."""
    out = {}
    for k, v in FUNCTIONS.items():
        out[k] = {
            "name":      v["name"],
            "latex":     v["latex"],
            "default_x0": v["default_x0"],
            "xlim":      v["xlim"],
            "ylim":      v["ylim"],
            "optimum":   v["optimum"],
            "f_opt":     v["f_opt"],
        }
    return out


@app.post("/optimize")
def run_optimize(req: OptimizeRequest):
    """Run steepest descent with the given method and return full path."""
    try:
        result = optimize(
            func_name   = req.func_name,
            method      = req.method,
            x0          = req.x0,
            max_iter    = req.max_iter,
            tol         = req.tol,
            armijo_s    = req.armijo_s,
            armijo_beta = req.armijo_beta,
            armijo_sigma= req.armijo_sigma,
            bt_beta     = req.bt_beta,
            bt_sigma    = req.bt_sigma,
            bb_alpha0   = req.bb_alpha0,
            el_alpha0   = req.el_alpha0,
        )
        return {
            "success":     result.success,
            "x_opt":       result.x_opt,
            "f_opt":       result.f_opt,
            "iterations":  result.iterations,
            "path":        result.path,
            "f_values":    result.f_values,
            "grad_norms":  result.grad_norms,
            "step_sizes":  result.step_sizes,
            "time_ms":     result.time_ms,
            "message":     result.message,
            "method":      result.method,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
def run_compare(req: CompareRequest):
    """Run multiple methods on the same function and return comparison."""
    results = {}
    for m in req.methods:
        try:
            r = optimize(
                func_name   = req.func_name,
                method      = m,
                x0          = req.x0,
                max_iter    = req.max_iter,
                tol         = req.tol,
            )
            results[m] = {
                "success":    r.success,
                "x_opt":      r.x_opt,
                "f_opt":      r.f_opt,
                "iterations": r.iterations,
                "path":       r.path,
                "f_values":   r.f_values,
                "grad_norms": r.grad_norms,
                "step_sizes": r.step_sizes,
                "time_ms":    r.time_ms,
                "message":    r.message,
            }
        except Exception as e:
            results[m] = {"success": False, "message": str(e), "iterations": 0}
    return results


@app.post("/contour")
def get_contour(req: ContourRequest):
    """Compute contour grid for plotting."""
    info = FUNCTIONS.get(req.func_name)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown function")
    f = info["f"]

    x_arr = np.linspace(req.xlim[0], req.xlim[1], req.resolution)
    y_arr = np.linspace(req.ylim[0], req.ylim[1], req.resolution)
    Z = np.zeros((req.resolution, req.resolution))
    for i, xi in enumerate(x_arr):
        for j, yj in enumerate(y_arr):
            try:
                val = f(np.array([xi, yj]))
                Z[j, i] = float(val) if np.isfinite(val) else 1e10
            except Exception:
                Z[j, i] = 1e10

    # clip extreme values for nice visualization
    p5, p95 = np.percentile(Z, 2), np.percentile(Z, 98)
    Z = np.clip(Z, p5, p95)

    return {
        "x": x_arr.tolist(),
        "y": y_arr.tolist(),
        "z": Z.tolist(),
    }
