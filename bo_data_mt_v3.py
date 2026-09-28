# bo_data_mt_v3.py
# Dataset building for multitask BO:
# metadata parsing, task-specific chemistry parsing, bounds, X/Y construction, task encoding
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reactant:
    name: str
    mw: float
    limiting_flag: int  # 1/0


@dataclass(frozen=True)
class TaskChemistry:
    """
    Task-specific chemistry metadata.

    task_key:
        Metadata key, e.g. "Task1", "Task2".
    task_label:
        User-facing task name, e.g. "naproxen_amide", "phe_lf".
    product_name / product_mw:
        Task-specific product identity. Falls back to global product metadata if empty.
    reactants:
        Task-specific Reactant objects. Names/MWs fall back to global reactants if empty.
        Limiting flags currently fall back to the global Reactant i Limiting Flag.
    """
    task_key: str
    task_label: str
    product_name: str
    product_mw: float
    reactants: List[Reactant]


@dataclass(frozen=True)
class BODesign:
    param_names: List[str]
    X: np.ndarray
    Y: np.ndarray
    bounds: np.ndarray
    simplex_n: int = 0
    objective_names: Optional[List[str]] = None


@dataclass(frozen=True)
class MultiTaskDesign:
    param_names: List[str]
    x_cols_with_task: List[str]         # param_names + [task_col]
    X_base: np.ndarray                  # shape (n, d)
    task_indices: np.ndarray            # shape (n, 1), integer-coded tasks
    X_aug: np.ndarray                   # shape (n, d+1), last col = task index
    Y: np.ndarray                       # shape (n, m)
    bounds: np.ndarray                  # shape (2, d)
    simplex_n: int = 0                  # if >0: first (simplex_n-1) params are free, last massfrac is computed
    objective_names: Optional[List[str]] = None
    task_col: str = "Task"
    task_to_index: Optional[Dict[str, int]] = None
    index_to_task: Optional[Dict[int, str]] = None
    task_chemistries: Optional[Dict[str, TaskChemistry]] = None


def _metadata_value(metadata: Dict[str, str], key: str, default: str = "") -> str:
    val = metadata.get(key, default)
    if val is None:
        return default
    return str(val).strip()


def _metadata_float(metadata: Dict[str, str], key: str, default: float = np.nan) -> float:
    val = metadata.get(key, "")
    if val in ("", None):
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def parse_reactants_from_metadata(metadata: Dict[str, str]) -> Tuple[float, str, float, int, List[Reactant]]:
    """
    Parse global/default chemistry metadata.

    This preserves the previous API:
        internal_volume, solvent_name, product_mw, n_reactants, reactants
    """
    internal_volume = float(metadata.get("Internal Volume (mL)", "0") or 0.0)
    solvent_name = str(metadata.get("Solvent Name", "") or "")
    product_mw = float(metadata.get("Product MW (g/mol)", "0") or 0.0)
    n_reactants = int(float(metadata.get("Number of Reactants", "0") or 0))

    reactants: List[Reactant] = []
    for i in range(1, n_reactants + 1):
        name = str(metadata.get(f"Reactant {i} Name", f"Reactant {i}") or f"Reactant {i}")
        mw_raw = metadata.get(f"Reactant {i} MW (g/mol)", "")
        if mw_raw in ("", None):
            raise ValueError(f"Missing molar mass for Reactant {i} in metadata.")
        mw = float(mw_raw)
        limiting = int(float(metadata.get(f"Reactant {i} Limiting Flag", "0") or 0))
        reactants.append(Reactant(name=name, mw=mw, limiting_flag=limiting))

    return internal_volume, solvent_name, product_mw, n_reactants, reactants


def parse_product_name_from_metadata(metadata: Dict[str, str]) -> str:
    """Return global product name, if present."""
    return _metadata_value(metadata, "Product Name", "product")


def parse_task_chemistries_from_metadata(
    metadata: Dict[str, str],
    *,
    max_tasks: int = 20,
) -> Dict[str, TaskChemistry]:
    """
    Parse task-specific product/reactant metadata written by the initializer.

    Expected keys:
        Task1, Task2, Task3, ...
        Task1 Product Name
        Task1 Product MW (g/mol)
        Task1 Reactant 1 Name
        Task1 Reactant 1 MW (g/mol)
        ...

    Empty task-specific fields fall back to global metadata:
        Product Name
        Product MW (g/mol)
        Reactant i Name
        Reactant i MW (g/mol)
        Reactant i Limiting Flag
    """
    _, _, global_product_mw, n_reactants, global_reactants = parse_reactants_from_metadata(metadata)
    global_product_name = parse_product_name_from_metadata(metadata)

    task_chemistries: Dict[str, TaskChemistry] = {}

    for t in range(1, max_tasks + 1):
        task_key = f"Task{t}"
        task_label = _metadata_value(metadata, task_key, "")
        if not task_label:
            continue

        product_name = _metadata_value(metadata, f"{task_key} Product Name", global_product_name)
        product_mw = _metadata_float(metadata, f"{task_key} Product MW (g/mol)", global_product_mw)

        reactants: List[Reactant] = []
        for i in range(1, n_reactants + 1):
            global_r = global_reactants[i - 1]
            r_name = _metadata_value(metadata, f"{task_key} Reactant {i} Name", global_r.name)
            r_mw = _metadata_float(metadata, f"{task_key} Reactant {i} MW (g/mol)", global_r.mw)
            reactants.append(
                Reactant(
                    name=r_name,
                    mw=float(r_mw),
                    limiting_flag=int(global_r.limiting_flag),
                )
            )

        task_chemistries[task_label] = TaskChemistry(
            task_key=task_key,
            task_label=task_label,
            product_name=product_name,
            product_mw=float(product_mw),
            reactants=reactants,
        )

    return task_chemistries


def get_task_chemistry(
    metadata: Dict[str, str],
    task_label: str,
    *,
    allow_global_fallback: bool = True,
) -> TaskChemistry:
    """
    Return task-specific chemistry metadata for a task label.

    If allow_global_fallback=True and task_label is not found, returns a TaskChemistry
    built from the global metadata using task_label as label.
    """
    task_label = str(task_label).strip()
    task_chemistries = parse_task_chemistries_from_metadata(metadata)
    if task_label in task_chemistries:
        return task_chemistries[task_label]

    if not allow_global_fallback:
        raise KeyError(f"Task chemistry not found in metadata: {task_label}")

    _, _, product_mw, _, reactants = parse_reactants_from_metadata(metadata)
    product_name = parse_product_name_from_metadata(metadata)
    return TaskChemistry(
        task_key="global",
        task_label=task_label or "global",
        product_name=product_name,
        product_mw=float(product_mw),
        reactants=reactants,
    )


def get_reactant_mws_for_task(metadata: Dict[str, str], task_label: str) -> np.ndarray:
    """Convenience helper: return reactant MWs for a given task label."""
    chem = get_task_chemistry(metadata, task_label)
    return np.array([r.mw for r in chem.reactants], dtype=float)


def get_product_mw_for_task(metadata: Dict[str, str], task_label: str) -> float:
    """Convenience helper: return product MW for a given task label."""
    return float(get_task_chemistry(metadata, task_label).product_mw)


def infer_bounds_from_last_row(
    df: pd.DataFrame,
    *,
    n_reactants: int,
    include_process: bool = True,
    debug: bool = False,
) -> Tuple[List[str], np.ndarray, int]:
    """
    Builds parameter names and bounds array from the last row of the experiment table.

    For mass-fraction mode, uses:
      - Reactant i MassFrac Range Bottom/Top
      - plus process ranges if present:
          Temperature Min/Max
          Time Min/Max
          Milling Load Min/Max
          Liquid Additive Min/Max

    Simplex handling:
      - We parameterize only the first (n_reactants-1) mass fractions
      - The last mass fraction is computed as 1 - sum(first n-1)
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    last = df.iloc[-1]

    param_names: List[str] = []
    bounds: List[Tuple[float, float]] = []
    simplex_n = int(n_reactants)

    # IMPORTANT: free simplex parameters are only the first (n_reactants - 1)
    for i in range(1, n_reactants):
        lo_col = f"Reactant {i} MassFrac Range Bottom"
        hi_col = f"Reactant {i} MassFrac Range Top"
        lb = float(last[lo_col])
        ub = float(last[hi_col])
        param_names.append(f"Reactant {i} MassFrac")
        bounds.append((lb, ub))

    if include_process:
        for nm, lo, hi in [
            ("Temperature (°C)", "Temperature Min (°C)", "Temperature Max (°C)"),
            ("Reaction Time (min)", "Time Min (min)", "Time Max (min)"),
            ("Milling Load (mg/mL)", "Milling Load Min (mg/mL)", "Milling Load Max (mg/mL)"),
            ("Liquid Additive (uL/mg)", "Liquid Additive Min (uL/mg)", "Liquid Additive Max (uL/mg)"),
        ]:
            if lo in df.columns and hi in df.columns:
                lb = float(last[lo])
                ub = float(last[hi])
                param_names.append(nm)
                bounds.append((lb, ub))

    bounds_arr = np.array(bounds, dtype=float).T  # (2, d)

    if debug:
        print("\n=========== BO BOUNDS ===========")
        for i, name in enumerate(param_names):
            lo = bounds_arr[0, i]
            hi = bounds_arr[1, i]
            print(f"{name:30s} [{lo:.6g}, {hi:.6g}]")
        print("=================================\n")

    return param_names, bounds_arr, simplex_n


def _build_fixed_map(max_reactants: int = 5) -> Dict[str, Tuple[str, str]]:
    fixed_map = {
        "Temperature (°C)": ("Temperature Min (°C)", "Temperature Max (°C)"),
        "Reaction Time (min)": ("Time Min (min)", "Time Max (min)"),
        "Milling Load (mg/mL)": ("Milling Load Min (mg/mL)", "Milling Load Max (mg/mL)"),
        "Liquid Additive (uL/mg)": ("Liquid Additive Min (uL/mg)", "Liquid Additive Max (uL/mg)"),
    }
    for i in range(1, max_reactants + 1):
        fixed_map[f"Reactant {i} MassFrac"] = (
            f"Reactant {i} MassFrac Range Bottom",
            f"Reactant {i} MassFrac Range Top",
        )
    return fixed_map


def _snap_fixed_variables(df: pd.DataFrame, X: np.ndarray, param_names: List[str], *, max_reactants: int = 5) -> np.ndarray:
    """
    If a variable is fixed (min == max), snap X to the fixed bound value.
    This avoids tiny experimental / rounding drift from entering the GP.
    """
    fixed_map = _build_fixed_map(max_reactants=max_reactants)

    for j, p in enumerate(param_names):
        mm = fixed_map.get(p)
        if mm is None:
            continue
        lo_col, hi_col = mm
        if lo_col not in df.columns or hi_col not in df.columns:
            continue

        lo = df[lo_col].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        hi = df[hi_col].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        fixed_mask = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
        if fixed_mask.any():
            X[fixed_mask, j] = lo[fixed_mask]

    return X


def encode_tasks(
    task_values: pd.Series,
    *,
    task_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encodes a task column into integer task indices.

    If task_order is provided, uses this exact order for stable task IDs.
    """
    s = task_values.astype("string").str.strip()

    if task_order is None:
        unique_tasks = [x for x in s.dropna().unique().tolist() if x != ""]
        unique_tasks = sorted(unique_tasks)
    else:
        unique_tasks = [str(x).strip() for x in task_order if str(x).strip() != ""]

    task_to_index = {t: i for i, t in enumerate(unique_tasks)}
    index_to_task = {i: t for t, i in task_to_index.items()}

    task_idx = np.full((len(s), 1), np.nan, dtype=float)
    for r, val in enumerate(s.tolist()):
        if val in task_to_index:
            task_idx[r, 0] = float(task_to_index[val])

    return task_idx, task_to_index, index_to_task


def task_order_from_metadata(metadata: Dict[str, str], *, max_tasks: int = 20) -> List[str]:
    """
    Return task labels ordered as Task1, Task2, Task3, ...
    Useful for stable task encoding.
    """
    labels: List[str] = []
    for t in range(1, max_tasks + 1):
        label = _metadata_value(metadata, f"Task{t}", "")
        if label:
            labels.append(label)
    return labels



def build_XY_from_df(
    df: pd.DataFrame,
    *,
    param_names: List[str],
    y_cols: List[str],
    dropna: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-task X/Y builder kept inside the consolidated multitask module.
    Fixed variables are snapped to Min when Min == Max.
    """
    X = df[param_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = df[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    X = _snap_fixed_variables(df, X, param_names)

    if dropna:
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X = X[mask]
        Y = Y[mask]

    if debug:
        print("\n================ BO DATA DEBUG ================")
        print("Parameters:", param_names)
        print("Targets:", y_cols)
        print("\nX shape:", X.shape)
        print(X)
        print("\nY shape:", Y.shape)
        print(Y)
        print("==============================================\n")

    return X, Y


def build_XYT_from_df(
    df: pd.DataFrame,
    *,
    param_names: List[str],
    y_cols: List[str],
    task_col: str = "Task",
    task_order: Optional[List[str]] = None,
    dropna: bool = True,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Builds:
      - X_base        shape (n, d)
      - task_indices  shape (n, 1)
      - Y             shape (n, m)

    The multitask model can then use:
      X_aug = np.hstack([X_base, task_indices])

    Rules:
      - If a parameter is fixed by bounds (Min == Max), X uses that fixed value.
      - task_col is required for multitask operation.
    """
    if task_col not in df.columns:
        raise ValueError(f"Task column not found: {task_col}")

    X = df[param_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y = df[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    X = _snap_fixed_variables(df, X, param_names)

    task_idx, task_to_index, index_to_task = encode_tasks(df[task_col], task_order=task_order)

    if dropna:
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1) & np.isfinite(task_idx).all(axis=1)
        X = X[mask]
        Y = Y[mask]
        task_idx = task_idx[mask]

    if debug:
        print("\n================ BO DATA DEBUG ================")
        print("Parameters:", param_names)
        print("Targets:", y_cols)
        print("Task column:", task_col)
        print("Task map:", task_to_index)

        print("\nX_base shape:", X.shape)
        print("X_base:")
        print(X)

        print("\nTask indices shape:", task_idx.shape)
        print("Task indices:")
        print(task_idx)

        print("\nY shape:", Y.shape)
        print("Y:")
        print(Y)

        print("==============================================\n")

    return X, task_idx, Y, task_to_index, index_to_task


def build_multitask_design(
    df: pd.DataFrame,
    *,
    metadata: Dict[str, str],
    y_cols: List[str],
    task_col: str = "Task",
    task_order: Optional[List[str]] = None,
    include_process: bool = True,
    dropna: bool = True,
    debug: bool = False,
) -> MultiTaskDesign:
    """
    High-level helper:
      1) infer bounds / param names from the last row
      2) build X_base, task_indices, Y
      3) concatenate X_aug = [X_base | task_idx]

    Returns a MultiTaskDesign dataclass.

    New in v2:
      - also returns task_chemistries containing task-specific product/reactant MWs.
    """
    _, _, _, n_reactants, _ = parse_reactants_from_metadata(metadata)

    if task_order is None:
        task_order = task_order_from_metadata(metadata)

    param_names, bounds, simplex_n = infer_bounds_from_last_row(
        df,
        n_reactants=n_reactants,
        include_process=include_process,
        debug=debug,
    )

    X_base, task_indices, Y, task_to_index, index_to_task = build_XYT_from_df(
        df,
        param_names=param_names,
        y_cols=y_cols,
        task_col=task_col,
        task_order=task_order,
        dropna=dropna,
        debug=debug,
    )

    X_aug = np.hstack([X_base, task_indices])
    task_chemistries = parse_task_chemistries_from_metadata(metadata)

    return MultiTaskDesign(
        param_names=param_names,
        x_cols_with_task=param_names + [task_col],
        X_base=X_base,
        task_indices=task_indices,
        X_aug=X_aug,
        Y=Y,
        bounds=bounds,
        simplex_n=simplex_n,
        objective_names=y_cols,
        task_col=task_col,
        task_to_index=task_to_index,
        index_to_task=index_to_task,
        task_chemistries=task_chemistries,
    )


def expand_simplex_X(X_free: np.ndarray, *, simplex_n: int) -> np.ndarray:
    """
    Given free parameters for a simplex of size simplex_n (using first simplex_n-1 dims),
    append the last mass fraction = 1 - sum(free).

    Returns expanded X with an additional column inserted at the end of the simplex block.
    """
    if simplex_n <= 0:
        return X_free

    k = simplex_n - 1
    if X_free.shape[1] < k:
        raise ValueError(f"X_free must have at least {k} cols for simplex_n={simplex_n}")

    first = X_free[:, :k]
    last = 1.0 - first.sum(axis=1, keepdims=True)
    rest = X_free[:, k:]
    return np.concatenate([first, last, rest], axis=1)


def compute_masses_moles_from_massfracs(
    *,
    internal_volume_ml: float,
    milling_load_mg_per_ml: float,
    liquid_additive_ul_per_mg: float,
    massfracs: np.ndarray,           # shape (n_reactants,)
    reactant_mws: np.ndarray,        # shape (n_reactants,)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Implements mass-fraction mass/moles computation:
      util_mass = internal_volume * milling_load
      masses = util_mass * massfracs
      moles = masses / reactant_mws
      total_input_mass = masses.sum() + liquid_additive_ul_per_mg * util_mass

    Returns:
      masses (mg or arbitrary units, consistent),
      moles (relative),
      total_input_mass
    """
    util_mass = internal_volume_ml * milling_load_mg_per_ml
    masses = util_mass * massfracs
    moles = masses / reactant_mws
    total_input_mass = masses.sum() + (liquid_additive_ul_per_mg * util_mass)
    return masses, moles, float(total_input_mass)


def compute_masses_moles_from_massfracs_for_task(
    *,
    metadata: Dict[str, str],
    task_label: str,
    internal_volume_ml: float,
    milling_load_mg_per_ml: float,
    liquid_additive_ul_per_mg: float,
    massfracs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Task-aware version of compute_masses_moles_from_massfracs.
    Uses task-specific reactant MWs from metadata.
    """
    reactant_mws = get_reactant_mws_for_task(metadata, task_label)
    return compute_masses_moles_from_massfracs(
        internal_volume_ml=internal_volume_ml,
        milling_load_mg_per_ml=milling_load_mg_per_ml,
        liquid_additive_ul_per_mg=liquid_additive_ul_per_mg,
        massfracs=massfracs,
        reactant_mws=reactant_mws,
    )
