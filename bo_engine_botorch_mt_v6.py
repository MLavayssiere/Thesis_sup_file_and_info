# bo_engine_botorch_mt_v6.py
# Consolidated single-task + multitask BoTorch engine
# BoTorch GP fitting + single-point LogEI suggestion + batch q-LogEI suggestion (sampled candidates)
# Supports a "simplex" parameterization for mass fractions:
#   - simplex_n = number of reactants N
#   - Only the first (N-1) mass fractions are optimized as free parameters.
#   - The last mass fraction is implied as 1 - sum(free).
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, ScalarizedPosteriorTransform
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex


@dataclass(frozen=True)
class FitResult:
    model: SingleTaskGP
    mll: ExactMarginalLogLikelihood


def _to_torch(X: np.ndarray, *, dtype=torch.double, device=None) -> torch.Tensor:
    t = torch.as_tensor(X, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _sample_bounded_simplex_free(
    *,
    n_candidates: int,
    simplex_n: int,
    simplex_bounds: np.ndarray | None,
    dtype=torch.double,
    device=None,
    oversample_factor: int = 8,
    max_rounds: int = 20,
) -> torch.Tensor:
    """
    Sample full N-simplex points satisfying individual mass-fraction bounds,
    then return only the first N-1 free variables.

    simplex_bounds: optional array shape (2, N), containing bounds for all N reactants.
    """
    N = int(simplex_n)
    if N <= 1:
        raise ValueError("simplex_n must be > 1 for simplex sampling.")

    if simplex_bounds is None:
        lo = torch.zeros(N, dtype=dtype, device=device)
        hi = torch.ones(N, dtype=dtype, device=device)
    else:
        sb = torch.as_tensor(simplex_bounds, dtype=dtype, device=device)
        if sb.shape[0] != 2 and sb.shape[1] == 2:
            sb = sb.transpose(0, 1)
        if sb.shape != (2, N):
            raise ValueError(f"simplex_bounds must have shape (2,{N}). Got {tuple(sb.shape)}")
        lo = sb[0]
        hi = sb[1]

    if torch.any(lo < -1e-12) or torch.any(hi > 1 + 1e-12) or torch.any(lo > hi):
        raise ValueError("Invalid simplex bounds: expected 0 <= lo <= hi <= 1.")
    if lo.sum() > 1 + 1e-12 or hi.sum() < 1 - 1e-12:
        raise ValueError("Infeasible simplex bounds: sum(lo) > 1 or sum(hi) < 1.")

    accepted = []
    need = int(n_candidates)

    for _ in range(max_rounds):
        n_draw = max(need * int(oversample_factor), need, 512)
        # BoTorch sample_simplex signature is sample_simplex(d, n)
        full = sample_simplex(N, n_draw, dtype=dtype, device=device)  # (n_draw, N)
        mask = (full >= lo.unsqueeze(0) - 1e-12).all(dim=1) & (full <= hi.unsqueeze(0) + 1e-12).all(dim=1)
        good = full[mask]
        if good.numel() > 0:
            accepted.append(good)
            got = sum(a.shape[0] for a in accepted)
            if got >= n_candidates:
                full_ok = torch.cat(accepted, dim=0)[:n_candidates]
                return full_ok[:, :N - 1]
        need = n_candidates - sum(a.shape[0] for a in accepted)

    raise RuntimeError(
        "Could not sample enough feasible simplex candidates. "
        "Check Reactant MassFrac Range Bottom/Top constraints."
    )



def fit_single_task_gp(X: np.ndarray, Y: np.ndarray, *, device=None) -> FitResult:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    X_t = _to_torch(X, device=device)
    Y_t = _to_torch(Y, device=device)

    model = SingleTaskGP(
        X_t,
        Y_t,
        input_transform=Normalize(d=X_t.shape[-1]),
        outcome_transform=Standardize(m=Y_t.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return FitResult(model=model, mll=mll)


def suggest_next_ei(
    fit: FitResult,
    *,
    bounds: np.ndarray,         # (2, d) numpy
    best_f: float,
    n_candidates: int = 4096,
    simplex_n: int = 0,         # number of reactants N (free dims are N-1)
    simplex_bounds: np.ndarray | None = None,  # optional (2,N) bounds for all mass fractions
    device=None,
) -> np.ndarray:
    """
    Suggest a single next point using analytic LogEI.

    Candidate generation:
      - Box variables: Sobol within bounds
      - If simplex_n>1: treat it as number of reactants N; sample N-simplex then drop last -> free vars sum<=1
    """
    model = fit.model
    model.eval()

    bounds = np.asarray(bounds, dtype=float)
    bounds_t = _to_torch(bounds, device=device)  # (2,d)
    if bounds_t.shape[0] != 2 and bounds_t.shape[1] == 2:
        bounds_t = bounds_t.transpose(0, 1)
    if bounds_t.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2,d). Got {tuple(bounds_t.shape)}")

    d_total = bounds_t.shape[1]

    # --- candidate generation ---
    if simplex_n and int(simplex_n) > 1:
        N = int(simplex_n)
        n_free = N - 1
        if d_total < n_free:
            raise ValueError(f"bounds has d={d_total} dims but needs at least {n_free} free simplex dims for N={N}")

        d_proc = d_total - n_free

        # Sample full simplex in N dims (sum=1), then drop last component -> free vars sum<=1
        mf = _sample_bounded_simplex_free(
            n_candidates=n_candidates,
            simplex_n=N,
            simplex_bounds=simplex_bounds,
            dtype=torch.double,
            device=bounds_t.device,
        )  # (n, N-1)

        if d_proc > 0:
            sobol = torch.quasirandom.SobolEngine(dimension=d_proc, scramble=True)
            U = sobol.draw(n_candidates).to(dtype=torch.double).to(bounds_t.device)  # (n, d_proc)
            lo = bounds_t[0, n_free:].unsqueeze(0)
            hi = bounds_t[1, n_free:].unsqueeze(0)
            proc = lo + (hi - lo) * U
            Xc = torch.cat([mf, proc], dim=1)
        else:
            Xc = mf
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=d_total, scramble=True)
        U = sobol.draw(n_candidates).to(dtype=torch.double).to(bounds_t.device)
        Xc = bounds_t[0] + (bounds_t[1] - bounds_t[0]) * U

    # --- acquisition scoring ---
    acq = LogExpectedImprovement(model=model, best_f=float(best_f))
    with torch.no_grad():
        vals = acq(Xc.unsqueeze(1)).squeeze(-1)  # (n,)
        idx = torch.argmax(vals).item()
        x_next = Xc[idx].cpu().numpy()

    return x_next


def suggest_q_logei(
    fit: FitResult,
    *,
    bounds: np.ndarray,          # (2, d) numpy
    best_f: float,
    q: int = 4,
    simplex_n: int = 0,          # number of reactants N (free dims are N-1)
    simplex_bounds: np.ndarray | None = None,  # optional (2,N) bounds for all mass fractions
    mc_samples: int = 256,
    raw_samples: int = 4096,     # number of candidate batches to score
    device=None,
) -> np.ndarray:
    """
    Batch suggestion using q-LogEI (MC), scored over many candidate *batches*.

    Returns: numpy array of shape (q, d)
    """
    model = fit.model
    model.eval()

    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1")

    bounds = np.asarray(bounds, dtype=float)
    bounds_t = _to_torch(bounds, device=device)  # (2,d)
    if bounds_t.shape[0] != 2 and bounds_t.shape[1] == 2:
        bounds_t = bounds_t.transpose(0, 1)
    if bounds_t.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2,d). Got {tuple(bounds_t.shape)}")

    d = bounds_t.shape[1]
    n_batches = int(raw_samples)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]))
    acq = qLogExpectedImprovement(model=model, best_f=float(best_f), sampler=sampler)

    # Build candidate batches Xcand: (n_batches, q, d)
    if simplex_n and int(simplex_n) > 1:
        N = int(simplex_n)
        n_free = N - 1
        if d < n_free:
            raise ValueError(f"bounds has d={d} dims but needs at least {n_free} free simplex dims for N={N}")

        d_proc = d - n_free

        # Sample full simplex in N dims then drop last -> free vars sum<=1
        mf_free_flat = _sample_bounded_simplex_free(
            n_candidates=q * n_batches,
            simplex_n=N,
            simplex_bounds=simplex_bounds,
            dtype=torch.double,
            device=bounds_t.device,
        )
        mf_free = mf_free_flat.view(n_batches, q, n_free)  # (n_batches, q, n_free)

        if d_proc > 0:
            sobol = torch.quasirandom.SobolEngine(dimension=d_proc, scramble=True)
            U = sobol.draw(q * n_batches).to(dtype=torch.double).to(bounds_t.device)  # (q*n_batches, d_proc)
            lo = bounds_t[0, n_free:].unsqueeze(0)
            hi = bounds_t[1, n_free:].unsqueeze(0)
            proc = (lo + (hi - lo) * U).view(n_batches, q, d_proc)
            Xcand = torch.cat([mf_free, proc], dim=-1)
        else:
            Xcand = mf_free
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        U = sobol.draw(q * n_batches).to(dtype=torch.double).to(bounds_t.device)  # (q*n_batches, d)
        X = bounds_t[0].unsqueeze(0) + (bounds_t[1] - bounds_t[0]).unsqueeze(0) * U
        Xcand = X.view(n_batches, q, d)

    with torch.no_grad():
        vals = acq(Xcand)  # (n_batches,)
        best = torch.argmax(vals).item()
        Xbest = Xcand[best]  # (q, d)

    return Xbest.detach().cpu().numpy()


# ---- Multitask engine section ----

# bo_engine_botorch_multitask_v1.py
# True multitask BO engine using BoTorch MultiTaskGP (ICM / dual-kernel GP)
# and task-conditional acquisition optimization by candidate sampling.
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls import ExactMarginalLogLikelihood


@dataclass(frozen=True)
class MultiTaskFitResult:
    model: MultiTaskGP
    mll: ExactMarginalLogLikelihood
    task_feature: int = -1


def _to_torch(X: np.ndarray, *, dtype=torch.double, device=None) -> torch.Tensor:
    t = torch.as_tensor(X, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _append_task_idx(X_base: np.ndarray, task_idx: int) -> np.ndarray:
    X_base = np.asarray(X_base, dtype=float)
    tcol = np.full((X_base.shape[0], 1), float(task_idx), dtype=float)
    return np.hstack([X_base, tcol])


def fit_multitask_gp(
    X_aug: np.ndarray,
    Y: np.ndarray,
    *,
    task_feature: int = -1,
    rank: Optional[int] = None,
    device=None,
) -> MultiTaskFitResult:
    """
    Fits a true multitask GP using BoTorch MultiTaskGP (ICM kernel).

    Parameters
    ----------
    X_aug : array, shape (n, d+1)
        Training inputs with the task index already appended as one column.
    Y : array, shape (n,) or (n, 1)
        Scalar observations.
    task_feature : int
        Column index of the task feature in X_aug. Default: last column.
    rank : optional int
        Rank of the task covariance. If None, BoTorch chooses a default / full rank.
    """
    X_aug = np.asarray(X_aug, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]

    X_t = _to_torch(X_aug, device=device)
    Y_t = _to_torch(Y, device=device)

    model = MultiTaskGP(
        train_X=X_t,
        train_Y=Y_t,
        task_feature=task_feature,
        rank=rank,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return MultiTaskFitResult(model=model, mll=mll, task_feature=task_feature)


def best_f_for_task(
    Y: np.ndarray,
    task_indices: np.ndarray,
    *,
    task_idx: int,
) -> float:
    """
    Compute best_f using only the selected task.
    """
    Y = np.asarray(Y, dtype=float).reshape(-1)
    task_indices = np.asarray(task_indices, dtype=float).reshape(-1)
    mask = np.isfinite(Y) & np.isfinite(task_indices) & (task_indices == float(task_idx))
    if not mask.any():
        raise ValueError(f"No finite observations found for task_idx={task_idx}.")
    return float(np.nanmax(Y[mask]))



def _scalar_output_transform(*, device=None):
    """
    MultiTaskGP advertises multiple outputs, so analytic acquisitions require
    a posterior_transform. However, when querying with a fixed task feature,
    the posterior can already be scalar-output. In that case the correct
    scalarization weight has length 1, not number_of_tasks.
    """
    return ScalarizedPosteriorTransform(
        weights=torch.ones(1, dtype=torch.double, device=device)
    )


def _task_mc_objective(model, task_idx: int):
    """
    MC acquisitions sometimes receive multi-output samples, sometimes scalar samples,
    depending on how MultiTaskGP posterior is queried.

    If samples are scalar-output (..., q), return them unchanged.
    If samples are multi-output (..., q, m), select the active task output.
    """
    def obj(samples, X=None):
        # Scalar-output posterior: samples shape usually sample_shape x batch_shape x q
        if samples.ndim >= 1:
            # Multi-output posterior: last dim is output dimension m
            n_outputs = int(getattr(model, "num_outputs", 1))
            if n_outputs > 1 and samples.shape[-1] == n_outputs:
                return samples[..., int(task_idx)]
        return samples

    return GenericMCObjective(obj)


def suggest_next_mt_ei(
    fit: MultiTaskFitResult,
    *,
    bounds: np.ndarray,         # (2, d_base)
    best_f: float,
    task_idx: int,
    n_candidates: int = 4096,
    simplex_n: int = 0,         # number of reactants N (free dims are N-1)
    simplex_bounds: np.ndarray | None = None,  # optional (2,N) bounds for all mass fractions
    device=None,
) -> np.ndarray:
    """
    Suggest a single next point for a selected task using analytic LogEI.

    Candidate generation:
      - box variables: Sobol within bounds
      - simplex variables: sample full N-simplex, then keep the first N-1 free vars
      - append fixed task_idx to every candidate before scoring

    Returns
    -------
    x_next_base : np.ndarray, shape (d_base,)
        Suggested base variables only (task is NOT included in the returned vector).
    """
    model = fit.model
    model.eval()

    bounds = np.asarray(bounds, dtype=float)
    bounds_t = _to_torch(bounds, device=device)  # (2, d_base)
    if bounds_t.shape[0] != 2 and bounds_t.shape[1] == 2:
        bounds_t = bounds_t.transpose(0, 1)
    if bounds_t.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2,d). Got {tuple(bounds_t.shape)}")

    d_base = bounds_t.shape[1]

    if simplex_n and int(simplex_n) > 1:
        N = int(simplex_n)
        n_free = N - 1
        if d_base < n_free:
            raise ValueError(f"bounds has d={d_base} dims but needs at least {n_free} free simplex dims for N={N}")

        d_proc = d_base - n_free

        mf = _sample_bounded_simplex_free(
            n_candidates=n_candidates,
            simplex_n=N,
            simplex_bounds=simplex_bounds,
            dtype=torch.double,
            device=bounds_t.device,
        )  # (n, N-1)

        if d_proc > 0:
            sobol = torch.quasirandom.SobolEngine(dimension=d_proc, scramble=True)
            U = sobol.draw(n_candidates).to(dtype=torch.double).to(bounds_t.device)
            lo = bounds_t[0, n_free:].unsqueeze(0)
            hi = bounds_t[1, n_free:].unsqueeze(0)
            proc = lo + (hi - lo) * U
            Xc_base = torch.cat([mf, proc], dim=1)
        else:
            Xc_base = mf
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=d_base, scramble=True)
        U = sobol.draw(n_candidates).to(dtype=torch.double).to(bounds_t.device)
        Xc_base = bounds_t[0] + (bounds_t[1] - bounds_t[0]) * U

    task_col = torch.full((Xc_base.shape[0], 1), float(task_idx), dtype=torch.double, device=Xc_base.device)
    Xc_aug = torch.cat([Xc_base, task_col], dim=1)

    acq = LogExpectedImprovement(
        model=model,
        best_f=float(best_f),
        posterior_transform=_scalar_output_transform(device=Xc_base.device),
    )
    with torch.no_grad():
        vals = acq(Xc_aug.unsqueeze(1)).squeeze(-1)  # (n,)
        idx = torch.argmax(vals).item()
        x_next = Xc_base[idx].cpu().numpy()

    return x_next


def suggest_q_logei_mt(
    fit: MultiTaskFitResult,
    *,
    bounds: np.ndarray,          # (2, d_base)
    best_f: float,
    task_idx: int,
    q: int = 4,
    simplex_n: int = 0,          # number of reactants N (free dims are N-1)
    simplex_bounds: np.ndarray | None = None,  # optional (2,N) bounds for all mass fractions
    mc_samples: int = 256,
    raw_samples: int = 4096,     # number of candidate batches to score
    device=None,
) -> np.ndarray:
    """
    Suggest a batch of q points for a selected task using q-LogEI.
    Uses candidate-batch sampling (robust on CPU, no optimize_acqf requirement).

    Returns
    -------
    Xbest_base : np.ndarray, shape (q, d_base)
        Suggested base variables only (task is NOT included in the returned array).
    """
    model = fit.model
    model.eval()

    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1")

    bounds = np.asarray(bounds, dtype=float)
    bounds_t = _to_torch(bounds, device=device)  # (2, d_base)
    if bounds_t.shape[0] != 2 and bounds_t.shape[1] == 2:
        bounds_t = bounds_t.transpose(0, 1)
    if bounds_t.shape[0] != 2:
        raise ValueError(f"bounds must have shape (2,d). Got {tuple(bounds_t.shape)}")

    d_base = bounds_t.shape[1]
    n_batches = int(raw_samples)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]))
    objective = _task_mc_objective(model, int(task_idx))
    acq = qLogExpectedImprovement(
        model=model,
        best_f=float(best_f),
        sampler=sampler,
        objective=objective,
    )

    if simplex_n and int(simplex_n) > 1:
        N = int(simplex_n)
        n_free = N - 1
        if d_base < n_free:
            raise ValueError(f"bounds has d={d_base} dims but needs at least {n_free} free simplex dims for N={N}")

        d_proc = d_base - n_free

        mf_free_flat = _sample_bounded_simplex_free(
            n_candidates=q * n_batches,
            simplex_n=N,
            simplex_bounds=simplex_bounds,
            dtype=torch.double,
            device=bounds_t.device,
        )
        mf_free = mf_free_flat.view(n_batches, q, n_free)  

        if d_proc > 0:
            sobol = torch.quasirandom.SobolEngine(dimension=d_proc, scramble=True)
            U = sobol.draw(q * n_batches).to(dtype=torch.double).to(bounds_t.device)
            lo = bounds_t[0, n_free:].unsqueeze(0)
            hi = bounds_t[1, n_free:].unsqueeze(0)
            proc = (lo + (hi - lo) * U).view(n_batches, q, d_proc)
            Xcand_base = torch.cat([mf_free, proc], dim=-1)
        else:
            Xcand_base = mf_free
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=d_base, scramble=True)
        U = sobol.draw(q * n_batches).to(dtype=torch.double).to(bounds_t.device)
        X = bounds_t[0].unsqueeze(0) + (bounds_t[1] - bounds_t[0]).unsqueeze(0) * U
        Xcand_base = X.view(n_batches, q, d_base)

    task_block = torch.full((n_batches, q, 1), float(task_idx), dtype=torch.double, device=Xcand_base.device)
    Xcand_aug = torch.cat([Xcand_base, task_block], dim=-1)

    with torch.no_grad():
        vals = acq(Xcand_aug)  # (n_batches,)
        best = torch.argmax(vals).item()
        Xbest = Xcand_base[best]  # (q, d_base)

    return Xbest.detach().cpu().numpy()
