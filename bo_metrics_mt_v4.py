# bo_metrics_mt_v4.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reactant:
    name: str
    mw: float
    limiting_flag: int  # 1/0


@dataclass(frozen=True)
class TaskChemistry:
    task_key: str
    task_label: str
    product_name: str
    product_mw: float
    reactants: List[Reactant]


def _meta_str(metadata: Dict[str, str], key: str, default: str = "") -> str:
    val = metadata.get(key, default)
    if val is None:
        return default
    return str(val).strip()


def _meta_float(metadata: Dict[str, str], key: str, default: float = np.nan) -> float:
    val = metadata.get(key, "")
    if val in ("", None):
        return float(default)
    try:
        return float(val)
    except Exception:
        return float(default)


def parse_reactants_from_metadata(metadata: Dict[str, str]) -> Tuple[float, float, int, List[Reactant]]:
    """
    Parse global/default chemistry metadata.

    Preserves the previous API:
        internal_volume, product_mw, n_reactants, reactants
    """
    internal_volume = float(metadata.get("Internal Volume (mL)", "0") or 0.0)
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

    return internal_volume, product_mw, n_reactants, reactants


def parse_product_name_from_metadata(metadata: Dict[str, str]) -> str:
    return _meta_str(metadata, "Product Name", "product")


def parse_task_chemistries_from_metadata(
    metadata: Dict[str, str],
    *,
    max_tasks: int = 20,
) -> Dict[str, TaskChemistry]:
    """
    Parse task-specific product/reactant metadata.

    Expected optional keys:
        Task1, Task2, ...
        Task1 Product Name
        Task1 Product MW (g/mol)
        Task1 Reactant 1 Name
        Task1 Reactant 1 MW (g/mol)
        ...

    Empty/missing task-specific fields fall back to global metadata.
    """
    _, global_product_mw, n_reactants, global_reactants = parse_reactants_from_metadata(metadata)
    global_product_name = parse_product_name_from_metadata(metadata)

    task_chemistries: Dict[str, TaskChemistry] = {}

    for t in range(1, max_tasks + 1):
        task_key = f"Task{t}"
        task_label = _meta_str(metadata, task_key, "")
        if not task_label:
            continue

        product_name = _meta_str(metadata, f"{task_key} Product Name", global_product_name)
        product_mw = _meta_float(metadata, f"{task_key} Product MW (g/mol)", global_product_mw)

        reactants: List[Reactant] = []
        for i in range(1, n_reactants + 1):
            global_r = global_reactants[i - 1]
            r_name = _meta_str(metadata, f"{task_key} Reactant {i} Name", global_r.name)
            r_mw = _meta_float(metadata, f"{task_key} Reactant {i} MW (g/mol)", global_r.mw)
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
    Return task-specific chemistry metadata for a row/task.

    If the task is not found and allow_global_fallback=True, returns global chemistry.
    """
    task_label = str(task_label or "").strip()
    task_chemistries = parse_task_chemistries_from_metadata(metadata)

    if task_label in task_chemistries:
        return task_chemistries[task_label]

    if not allow_global_fallback:
        raise KeyError(f"Task chemistry not found in metadata: {task_label}")

    _, product_mw, _, reactants = parse_reactants_from_metadata(metadata)
    product_name = parse_product_name_from_metadata(metadata)
    return TaskChemistry(
        task_key="global",
        task_label=task_label or "global",
        product_name=product_name,
        product_mw=float(product_mw),
        reactants=reactants,
    )


def _as_float(x) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _normalize_conversion(conv: float) -> float:
    """Accept 0-100 or 0-1, return 0-1."""
    if not np.isfinite(conv):
        return float("nan")
    if conv > 1.0:
        return float(conv / 100.0)
    return float(conv)


def compute_row_derived(
    row: pd.Series,
    *,
    internal_volume_ml: float,
    product_mw: float,
    reactants: List[Reactant],
    n_reactants: int,
    massfrac_cols: List[str],
    milling_col: str,
    addliq_col: str,
    conversion_col: Optional[str],
    time_col: Optional[str],
) -> Dict[str, float]:
    milling = _as_float(row.get(milling_col, np.nan))
    addliq = _as_float(row.get(addliq_col, 0.0))
    rxn_time_min = _as_float(row.get(time_col, np.nan)) if time_col else float("nan")

    util_mass = internal_volume_ml * milling if np.isfinite(milling) else float("nan")

    mfs = np.array([_as_float(row.get(c, np.nan)) for c in massfrac_cols], dtype=float)
    masses = util_mass * mfs if np.isfinite(util_mass) else np.full_like(mfs, np.nan)

    mws = np.array([r.mw for r in reactants], dtype=float)
    moles = masses / mws  # masses mg, mw g/mol -> moles in 1e-3 mol (consistent)

    limiting_mask = np.array([r.limiting_flag == 1 for r in reactants], dtype=bool)
    limiting_moles = moles[limiting_mask]
    limiting_moles = limiting_moles[np.isfinite(limiting_moles) & (limiting_moles > 0)]
    n_moles = float(np.min(limiting_moles)) if limiting_moles.size else float("nan")

    # equivalents: lowest limiting reactant gets 1.0
    equiv = (moles / n_moles) if np.isfinite(n_moles) and n_moles > 0 else np.full_like(moles, np.nan)

    # product mass if conversion present
    mass_of_product = float("nan")
    conv_frac = float("nan")
    if conversion_col and conversion_col in row.index:
        conv = _as_float(row.get(conversion_col, np.nan))
        conv_frac = _normalize_conversion(conv)
        if np.isfinite(conv_frac) and np.isfinite(n_moles) and n_moles > 0 and np.isfinite(product_mw) and product_mw > 0:
            mass_of_product = float(conv_frac * n_moles * product_mw)  # mg under current unit convention

    total_input_mass = float(
        np.nansum(masses)
        + (addliq * util_mass if np.isfinite(util_mass) and np.isfinite(addliq) else 0.0)
    )

    pmi = float("nan")
    if np.isfinite(mass_of_product) and mass_of_product > 0 and np.isfinite(total_input_mass) and total_input_mass >= 0:
        pmi = float(total_input_mass / mass_of_product)

    # Productivity: mg/min in current units (mass_of_product mg, time min)
    productivity = float("nan")
    if np.isfinite(mass_of_product) and mass_of_product > 0 and np.isfinite(rxn_time_min) and rxn_time_min > 0:
        productivity = float(mass_of_product / rxn_time_min)

    # Target: 0.5*(NMR_conversion/100) + 0.5*(1/PMI)
    target = float("nan")
    if np.isfinite(conv_frac) and np.isfinite(pmi) and pmi > 0:
        target = float(0.5 * conv_frac + 0.5 * (1.0 / pmi))

    out: Dict[str, float] = {
        "n moles": n_moles,
        "util_mass": util_mass,
        "total_input_mass": total_input_mass,
        "mass_of_product": mass_of_product,
        "PMI": pmi,
        "Productivity": productivity,
        "target": target,
    }

    for i in range(1, n_reactants + 1):
        out[f"Reactant {i} Mass"] = float(masses[i - 1]) if i - 1 < len(masses) else float("nan")
        out[f"Reactant {i} Moles"] = float(moles[i - 1]) if i - 1 < len(moles) else float("nan")
        out[f"Reactant {i} Equiv"] = float(equiv[i - 1]) if i - 1 < len(equiv) else float("nan")

    return out


def refresh_table(
    metadata: Dict[str, str],
    df: pd.DataFrame,
    *,
    conversion_col: str = "NMR_conversion",
    time_col: str = "Reaction Time (min)",
    milling_col: str = "Milling Load (mg/mL)",
    addliq_col: str = "Liquid Additive (uL/mg)",
    massfrac_suffix: str = "MassFrac",
    task_col: str = "Task",
) -> pd.DataFrame:
    """
    Refresh derived columns.

    v3 change:
      If a Task column exists, each row uses task-specific product/reactant MWs
      from metadata:
          Task1 Product MW (g/mol)
          Task1 Reactant 1 MW (g/mol)
          ...
      Missing task-specific values fall back to the global metadata.
    """
    internal_volume, global_product_mw, n_reactants, global_reactants = parse_reactants_from_metadata(metadata)
    df2 = df.copy()

    massfrac_cols = [f"Reactant {i} {massfrac_suffix}" for i in range(1, n_reactants + 1)]
    required = (
        ["n moles", "PMI", "Productivity", "target", "mass_of_product", "util_mass", "total_input_mass"]
        + [f"Reactant {i} Mass" for i in range(1, n_reactants + 1)]
        + [f"Reactant {i} Moles" for i in range(1, n_reactants + 1)]
        + [f"Reactant {i} Equiv" for i in range(1, n_reactants + 1)]
    )
    df2 = ensure_columns(df2, required)

    conv_eff = conversion_col if conversion_col in df2.columns else None
    time_eff = time_col if time_col in df2.columns else None

    has_task = task_col in df2.columns
    task_chemistries = parse_task_chemistries_from_metadata(metadata) if has_task else {}

    for idx, row in df2.iterrows():
        if has_task:
            task_label = str(row.get(task_col, "") or "").strip()
            chem = task_chemistries.get(task_label)
            if chem is None:
                chem = get_task_chemistry(metadata, task_label, allow_global_fallback=True)
            product_mw = chem.product_mw
            reactants = chem.reactants
        else:
            product_mw = global_product_mw
            reactants = global_reactants

        derived = compute_row_derived(
            row,
            internal_volume_ml=internal_volume,
            product_mw=product_mw,
            reactants=reactants,
            n_reactants=n_reactants,
            massfrac_cols=massfrac_cols,
            milling_col=milling_col,
            addliq_col=addliq_col,
            conversion_col=conv_eff,
            time_col=time_eff,
        )

        for k, v in derived.items():
            if k not in df2.columns:
                df2[k] = np.nan
            # Prevent pandas FutureWarning when writing floats into an int column
            if k in df2.columns and not pd.api.types.is_float_dtype(df2[k]) and not pd.api.types.is_object_dtype(df2[k]):
                try:
                    df2[k] = df2[k].astype(float)
                except Exception:
                    pass
            df2.at[idx, k] = v

    return df2
