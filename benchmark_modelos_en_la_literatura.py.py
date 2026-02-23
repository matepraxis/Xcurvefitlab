

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Modelos


def model_logaritmico(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # y = a * ln(b*x + c) + d
    a, b, c, d = theta
    return a * np.log(b * x + c) + d

def model_polinomico(x: np.ndarray, theta: np.ndarray, grado: int) -> np.ndarray:
    # theta = [c0, c1, ..., c_grado] (orden descendente)
    y = np.zeros_like(x, dtype=float)
    for i in range(grado + 1):
        y += theta[i] * (x ** (grado - i))
    return y

def model_exponencial(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # y = a * exp(b*x) + c
    a, b, c = theta
    return a * np.exp(b * x) + c

def model_trigonometrico(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # y = a*sin(bx) + c*cos(dx) + e
    a, b, c, d, e = theta
    return a * np.sin(b * x) + c * np.cos(d * x) + e

def model_logistico(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # y = L / (1 + exp(-k(x-x0)))
    L, k, x0 = theta
    return L / (1.0 + np.exp(-k * (x - x0)))


MODEL_SPECS: Dict[str, Dict] = {
    "logaritmico":    {"p": 4, "fn": model_logaritmico},
    "exponencial":    {"p": 3, "fn": model_exponencial},
    "trigonometrico": {"p": 5, "fn": model_trigonometrico},
    "logistico":      {"p": 3, "fn": model_logistico},

}



# Métricas


def sse(y: np.ndarray, yhat: np.ndarray) -> float:
    r = y - yhat
    return float(np.dot(r, r))

def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return sse(y, yhat) / float(len(y))

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = sse(y, yhat)
    ss_tot = float(np.dot(y - np.mean(y), y - np.mean(y)))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot



# derivadas numéricas


def safe_model_eval(model: Callable, x: np.ndarray, theta: np.ndarray, **kwargs) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        yhat = model(x, theta, **kwargs) if kwargs else model(x, theta)
    yhat = np.asarray(yhat, dtype=float)
    return yhat

def residuals(model: Callable, x: np.ndarray, y: np.ndarray, theta: np.ndarray, **kwargs) -> np.ndarray:
    yhat = safe_model_eval(model, x, theta, **kwargs)
    r = y - yhat
    bad = ~np.isfinite(r)
    if np.any(bad):
        r = r.copy()
        r[bad] = 1e3  # penalización finita (residual grande)
    return r

def jacobian_fd(model: Callable, x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                eps: float = 1e-6, **kwargs) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    r0 = residuals(model, x, y, theta, **kwargs)
    n = len(r0)
    p = len(theta)
    J = np.zeros((n, p), dtype=float)

    for j in range(p):
        step = eps * (1.0 + abs(theta[j]))
        th_plus = theta.copy();  th_plus[j] += step
        th_minus = theta.copy(); th_minus[j] -= step
        r_plus = residuals(model, x, y, th_plus, **kwargs)
        r_minus = residuals(model, x, y, th_minus, **kwargs)
        J[:, j] = (r_plus - r_minus) / (2.0 * step)
    return J



# Optimizadores


@dataclass
class FitResult:
    method: str
    theta: np.ndarray
    mse: float
    r2: float
    iters: int
    elapsed_s: float
    converged: bool


def _solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def fit_gauss_newton(model: Callable, x: np.ndarray, y: np.ndarray, theta0: np.ndarray,
                     max_iter: int = 200, tol: float = 1e-8, **kwargs) -> FitResult:
    t0 = time.time()
    theta = np.asarray(theta0, dtype=float).copy()
    converged = False
    last_sse = None

    for k in range(1, max_iter + 1):
        r = residuals(model, x, y, theta, **kwargs)
        J = jacobian_fd(model, x, y, theta, **kwargs)
        A = J.T @ J
        g = J.T @ r
        delta = _solve_linear(A, g)
        theta_new = theta - delta

        yhat_new = safe_model_eval(model, x, theta_new, **kwargs)
        cur_sse = sse(y, yhat_new)

        if last_sse is not None and abs(last_sse - cur_sse) <= tol * (1 + last_sse):
            converged = True
            theta = theta_new
            break

        theta = theta_new
        last_sse = cur_sse

    yhat = safe_model_eval(model, x, theta, **kwargs)
    return FitResult("Gauss-Newton", theta, mse(y, yhat), r2_score(y, yhat), k, time.time()-t0, converged)


def fit_levenberg_marquardt(model: Callable, x: np.ndarray, y: np.ndarray, theta0: np.ndarray,
                            max_iter: int = 400, tol: float = 1e-10,
                            lam0: float = 1e-2, lam_factor: float = 10.0, **kwargs) -> FitResult:
    t0 = time.time()
    theta = np.asarray(theta0, dtype=float).copy()
    lam = lam0
    converged = False

    yhat = safe_model_eval(model, x, theta, **kwargs)
    best_sse = sse(y, yhat)

    for k in range(1, max_iter + 1):
        r = residuals(model, x, y, theta, **kwargs)
        J = jacobian_fd(model, x, y, theta, **kwargs)

        A = J.T @ J
        g = J.T @ r

        delta = _solve_linear(A + lam * np.eye(A.shape[0]), g)
        theta_try = theta - delta

        yhat_try = safe_model_eval(model, x, theta_try, **kwargs)
        sse_try = sse(y, yhat_try)

        if sse_try < best_sse:
            if abs(best_sse - sse_try) <= tol * (1 + best_sse):
                theta = theta_try
                best_sse = sse_try
                converged = True
                break
            theta = theta_try
            best_sse = sse_try
            lam = max(lam / lam_factor, 1e-12)
        else:
            lam = min(lam * lam_factor, 1e12)

    yhat = safe_model_eval(model, x, theta, **kwargs)
    return FitResult("Levenberg-Marquardt", theta, mse(y, yhat), r2_score(y, yhat), k, time.time()-t0, converged)


def _grad_sse(model: Callable, x: np.ndarray, y: np.ndarray, theta: np.ndarray, **kwargs) -> np.ndarray:
    r = residuals(model, x, y, theta, **kwargs)
    J = jacobian_fd(model, x, y, theta, **kwargs)
    return J.T @ r  # grad(0.5*SSE) = J^T r


def hessian_fd_from_grad(model: Callable, x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                         eps: float = 1e-5, **kwargs) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    p = len(theta)
    H = np.zeros((p, p), dtype=float)

    for j in range(p):
        step = eps * (1.0 + abs(theta[j]))
        th_plus = theta.copy();  th_plus[j] += step
        th_minus = theta.copy(); th_minus[j] -= step
        g_plus = _grad_sse(model, x, y, th_plus, **kwargs)
        g_minus = _grad_sse(model, x, y, th_minus, **kwargs)
        H[:, j] = (g_plus - g_minus) / (2.0 * step)

    return 0.5 * (H + H.T)


def fit_newton_raphson(model: Callable, x: np.ndarray, y: np.ndarray, theta0: np.ndarray,
                       max_iter: int = 80, tol: float = 1e-10, damping: float = 1.0, **kwargs) -> FitResult:
    t0 = time.time()
    theta = np.asarray(theta0, dtype=float).copy()
    converged = False

    yhat = safe_model_eval(model, x, theta, **kwargs)
    best_sse = sse(y, yhat)

    for k in range(1, max_iter + 1):
        g = _grad_sse(model, x, y, theta, **kwargs)
        H = hessian_fd_from_grad(model, x, y, theta, **kwargs)

        delta = _solve_linear(H, g)
        theta_try = theta - damping * delta

        yhat_try = safe_model_eval(model, x, theta_try, **kwargs)
        sse_try = sse(y, yhat_try)

        bt = 0
        local_damping = damping
        while sse_try > best_sse and bt < 10:
            local_damping *= 0.5
            theta_try = theta - local_damping * delta
            yhat_try = safe_model_eval(model, x, theta_try, **kwargs)
            sse_try = sse(y, yhat_try)
            bt += 1

        if abs(best_sse - sse_try) <= tol * (1 + best_sse):
            theta = theta_try
            best_sse = sse_try
            converged = True
            break

        theta = theta_try
        best_sse = sse_try

    yhat = safe_model_eval(model, x, theta, **kwargs)
    return FitResult("Newton-Raphson (SSE)", theta, mse(y, yhat), r2_score(y, yhat), k, time.time()-t0, converged)


def fit_gradient_descent(model: Callable, x: np.ndarray, y: np.ndarray, theta0: np.ndarray,
                         lr: float = 1e-3, max_iter: int = 6000, tol: float = 1e-10, **kwargs) -> FitResult:
    t0 = time.time()
    theta = np.asarray(theta0, dtype=float).copy()
    converged = False

    yhat = safe_model_eval(model, x, theta, **kwargs)
    best_sse = sse(y, yhat)

    for k in range(1, max_iter + 1):
        g = _grad_sse(model, x, y, theta, **kwargs)
        theta_try = theta - lr * g

        yhat_try = safe_model_eval(model, x, theta_try, **kwargs)
        sse_try = sse(y, yhat_try)

        if sse_try < best_sse:
            if abs(best_sse - sse_try) <= tol * (1 + best_sse):
                theta = theta_try
                best_sse = sse_try
                converged = True
                break
            theta = theta_try
            best_sse = sse_try
            lr *= 1.02
        else:
            lr *= 0.5

    yhat = safe_model_eval(model, x, theta, **kwargs)
    return FitResult("Gradiente (SSE)", theta, mse(y, yhat), r2_score(y, yhat), k, time.time()-t0, converged)


# polinomios


def fit_poly_ols(x: np.ndarray, y: np.ndarray, grado: int) -> FitResult:
    t0 = time.time()
    X = np.vander(x, N=grado + 1, increasing=False)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ theta
    return FitResult(f"OLS polinomio grado {grado}", theta, mse(y, yhat), r2_score(y, yhat), 1, time.time()-t0, True)


# CSV 


def load_csv(csv_path: str, x_col: Optional[str], y_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if x_col is None:
        x_col = df.columns[0]
    if y_col is None:
        y_col = df.columns[1]
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    return x, y

def random_theta(p: int, low: float = -10.0, high: float = 10.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.uniform(low, high, size=p)



# Gráfica puntos 


def predict(model_name: str, x: np.ndarray, theta: np.ndarray, grado: Optional[int] = None) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if model_name == "polinomico":
        if grado is None:
            raise ValueError("Para polinomico debes indicar grado.")
        return model_polinomico(x, theta, grado=grado)
    return MODEL_SPECS[model_name]["fn"](x, theta)

def save_plot_compare(model_name: str, x: np.ndarray, y: np.ndarray, results: List[FitResult],
                      grado: Optional[int] = None, out_file: str = "comparacion.png") -> None:
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_fine = np.linspace(np.min(x), np.max(x), 1200)

    plt.figure(figsize=(16, 7))
    plt.scatter(x, y, s=22) 

    for r in results:
        yhat = predict(model_name, x_fine, r.theta, grado=grado)
        m = np.isfinite(yhat)
        if np.any(m):
            plt.plot(x_fine[m], yhat[m], linewidth=3, label=r.method)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Comparación (dataset real): ajustes {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()


# Iniciador del benchmark


def run_one_fit(model_name: str, x: np.ndarray, y: np.ndarray, grado: Optional[int],
                theta0: np.ndarray, method: str) -> FitResult:
    if model_name == "polinomico":
        if grado is None:
            raise ValueError("Para polinomico debes indicar --grado.")
        if method == "OLS":
            return fit_poly_ols(x, y, grado)

        model = lambda xx, th: model_polinomico(xx, th, grado=grado)

        if method == "GN":
            return fit_gauss_newton(model, x, y, theta0, max_iter=200)
        if method == "LM":
            return fit_levenberg_marquardt(model, x, y, theta0, max_iter=400)
        if method == "NR":
            return fit_newton_raphson(model, x, y, theta0, max_iter=80)
        if method == "GD":
            return fit_gradient_descent(model, x, y, theta0, lr=1e-4, max_iter=8000)
        raise ValueError("Método no soportado.")
    else:
        spec = MODEL_SPECS[model_name]
        model = spec["fn"]

        if method == "GN":
            return fit_gauss_newton(model, x, y, theta0, max_iter=200)
        if method == "LM":
            return fit_levenberg_marquardt(model, x, y, theta0, max_iter=400)
        if method == "NR":
            return fit_newton_raphson(model, x, y, theta0, max_iter=80)
        if method == "GD":
            return fit_gradient_descent(model, x, y, theta0, lr=1e-3, max_iter=6000)
        raise ValueError("Método no soportado.")



def main() -> None:
    ap = argparse.ArgumentParser()

    # Defaults para tu caso (data.csv, Tiempo, Var6, trigonometrico)
    ap.add_argument("--csv", default="data.csv", help="Ruta del CSV (default: data.csv).")
    ap.add_argument("--x", default="Tiempo", help="Columna X (default: Tiempo).")
    ap.add_argument("--y", default="Var6", help="Columna Y (default: Var6).")
    ap.add_argument(
        "--model",
        default="trigonometrico",
        choices=["logaritmico", "exponencial", "trigonometrico", "logistico", "polinomico"],
        help="Tipo de modelo (default: trigonometrico)."
    )
    ap.add_argument("--grado", type=int, default=None, help="Grado si model=polinomico.")
    ap.add_argument("--restarts", type=int, default=25, help="Número de reinicios aleatorios (GN/LM/NR/GD).")
    ap.add_argument("--seed", type=int, default=123, help="Semilla.")
    ap.add_argument("--out", default="resultados_fits.csv", help="CSV de salida con métricas.")

    # Plot ON por defecto (Python 3.9+)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="Guardar la gráfica comparativa (default: True).")
    ap.add_argument("--plot_file", default="comparacion.png", help="Nombre del PNG (default: comparacion.png).")

    args = ap.parse_args()
    print("SEED USADA:", args.seed)

    x, y = load_csv(args.csv, args.x, args.y)
    rng = np.random.default_rng(args.seed)

    results: List[FitResult] = []

    # OLS para polinomio (si aplica)
    if args.model == "polinomico":
        if args.grado is None:
            raise SystemExit("ERROR: Si model=polinomico debes indicar --grado (ej: --grado 5).")
        results.append(run_one_fit("polinomico", x, y, args.grado, theta0=np.zeros(args.grado + 1), method="OLS"))

    methods = ["GN", "LM", "NR", "GD"]
    p = (args.grado + 1) if args.model == "polinomico" else MODEL_SPECS[args.model]["p"]

    for m in methods:
        best: Optional[FitResult] = None
        for _ in range(max(args.restarts, 1)):
            theta0 = random_theta(p, rng=rng)
            r = run_one_fit(args.model, x, y, args.grado, theta0=theta0, method=m)
            if best is None or r.mse < best.mse:
                best = r
        results.append(best)

    # Tabla a CSV
    df = pd.DataFrame([{
        "method": r.method,
        "mse": r.mse,
        "r2": r.r2,
        "iters": r.iters,
        "elapsed_s": r.elapsed_s,
        "converged": r.converged,
        "theta": r.theta.tolist(),
    } for r in results]).sort_values("mse")

    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nGuardado: {args.out}")

    # Gráfica (como tu ejemplo)
    if args.plot:
        save_plot_compare(args.model, x, y, results, grado=args.grado, out_file=args.plot_file)
        print(f"Imagen guardada: {args.plot_file}")


if __name__ == "__main__":
    main()
