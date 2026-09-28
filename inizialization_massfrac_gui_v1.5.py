# -*- coding: utf-8 -*-
"""
Initialization / CSV builder (Mass-Fraction GUI) — repaired v1.5

Purpose
-------
Create a *two-block CSV* compatible with the BO GUI:
  - Block 1: metadata key/value pairs (2 columns)
  - Blank separator row
  - Block 2: rectangular table with header + initial rows

Design assumptions (matching your current BO GUI):
  - Reactants are "Reactant 1" .. "Reactant 5"
  - Active reactants: N = 2..5 (Number of Reactants)
  - Mass fractions are defined only for active reactants and sum to 1 across them.
  - BO parameterization later uses the first (N-1) mass fractions as free vars; last is implied.
  - This initializer writes *all* N mass fractions for clarity.

Notes
-----
- Masses are initialized using:
      util_mass = internal_volume * milling_load
      mass_i = util_mass * massfrac_i
- NMR / Productivity / PMI / target / mass_of_product are left blank (NaN) for the user / BO GUI.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ----------------------------
# Helpers
# ----------------------------
def _safe_float(s: str, default: float = np.nan) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default


def _safe_int(s: str, default: int | None = None) -> int | None:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return default


def _fmt(x) -> str:
    """Format numbers for CSV output (avoid trailing .0 where possible)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        if np.isnan(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        # keep a reasonable precision
        return f"{float(x):.16g}"
    return str(x)


def _sample_massfracs_with_bounds(
    mins: np.ndarray,
    maxs: np.ndarray,
    modes: List[str],
    *,
    max_tries: int = 20000,
) -> np.ndarray:
    """
    Sample a feasible mass fraction vector mf (len=N) such that:
      - fixed entries are fixed to mins[i] (== maxs[i])
      - range entries are within [mins[i], maxs[i]]
      - sum(mf) == 1
    Uses rejection sampling + renormalization. Works well for N<=5.
    """
    N = len(mins)
    fixed = np.zeros(N, dtype=float)
    var_idx = []
    for i in range(N):
        if modes[i] == "fixed":
            fixed[i] = float(mins[i])
        else:
            var_idx.append(i)

    remaining = 1.0 - float(fixed.sum())
    if remaining < -1e-12:
        raise ValueError("Fixed mass fractions sum to > 1.")
    if remaining < 1e-12 and len(var_idx) == 0:
        return fixed

    vmins = np.array([mins[i] for i in var_idx], dtype=float)
    vmaxs = np.array([maxs[i] for i in var_idx], dtype=float)

    # Quick feasibility check: can the variable part sum to remaining?
    if len(var_idx) > 0:
        if vmins.sum() - 1e-12 > remaining:
            raise ValueError("MassFrac mins are infeasible (sum of mins exceeds 1 - fixed_sum).")
        if vmaxs.sum() + 1e-12 < remaining:
            raise ValueError("MassFrac maxs are infeasible (sum of maxs below 1 - fixed_sum).")

    for _ in range(max_tries):
        mf = fixed.copy()
        if len(var_idx) == 0:
            # No variables, but remaining ~0 already handled; otherwise infeasible
            continue

        # sample raw values within bounds
        raw = np.array([random.uniform(vmins[j], vmaxs[j]) for j in range(len(var_idx))], dtype=float)

        s = raw.sum()
        if s <= 1e-12:
            continue

        # rescale to hit the required remaining sum
        scaled = raw * (remaining / s)

        # check bounds after scaling (this is the rejection part)
        if np.all(scaled >= vmins - 1e-12) and np.all(scaled <= vmaxs + 1e-12):
            for j, i in enumerate(var_idx):
                mf[i] = float(scaled[j])

            # numerical cleanup
            mf = np.clip(mf, 0.0, 1.0)
            mf = mf / mf.sum()  # enforce sum=1
            return mf

    raise ValueError("Could not sample a feasible mass fraction vector. Try widening bounds or reducing fixed values.")


# ----------------------------
# GUI
# ----------------------------
@dataclass
class ReactantSpec:
    name: str
    mw: float
    limiting_flag: int
    mode: str            # "range" or "fixed"
    mf_min: float
    mf_max: float


class InitializationMassFracGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Initialization — MassFrac CSV builder (v1.5)")
        self.geometry("1050x720")

        # Global vars
        self.num_of_reactants = tk.StringVar(value="3")
        self.n_initial_rows = tk.StringVar(value="5")

        self.internal_volume = tk.StringVar(value="2.0")
        self.solvent_name = tk.StringVar(value="xxx")
        self.product_name = tk.StringVar(value="amide")
        self.product_mw = tk.StringVar(value="360.0")
        self.task1 = tk.StringVar(value="task_A")
        self.task2 = tk.StringVar(value="")
        self.task3 = tk.StringVar(value="")

        # Optional task-specific product/reactant identities and MWs.
        # Empty task-specific fields fall back to the global values.
        self.task_product_name_vars = [[tk.StringVar(value="") for _ in range(1)] for _ in range(3)]
        self.task_product_mw_vars = [[tk.StringVar(value="") for _ in range(1)] for _ in range(3)]
        self.task_reactant_name_vars = [[tk.StringVar(value="") for _ in range(5)] for _ in range(3)]
        self.task_reactant_mw_vars = [[tk.StringVar(value="") for _ in range(5)] for _ in range(3)]

        self.temp_min_var = tk.StringVar(value="25")
        self.temp_max_var = tk.StringVar(value="125")
        self.time_min_var = tk.StringVar(value="5")
        self.time_max_var = tk.StringVar(value="60")
        self.milling_min_var = tk.StringVar(value="1.0")
        self.milling_max_var = tk.StringVar(value="1.0")
        self.addliq_min_var = tk.StringVar(value="0.0")
        self.addliq_max_var = tk.StringVar(value="0.0")

        self.reactant_entries: List[dict] = []

        self._build_ui()
        self._toggle_reactant_rows()

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Number of reactants (2–5):").grid(row=0, column=0, sticky="w", **pad)
        num_entry = ttk.Entry(top, textvariable=self.num_of_reactants, width=6)
        num_entry.grid(row=0, column=1, sticky="w", **pad)
        # Validate on focus-out / Return (so typing "4" does not become "34" -> clamped to 5)
        num_entry.bind("<Return>", lambda e: self._toggle_reactant_rows())
        num_entry.bind("<FocusOut>", lambda e: self._toggle_reactant_rows())
        ttk.Label(top, text="Initial rows:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.n_initial_rows, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Separator(self).pack(fill="x", padx=10, pady=6)

        meta = ttk.LabelFrame(self, text="Metadata")
        meta.pack(fill="x", padx=10, pady=6)

        ttk.Label(meta, text="Internal Volume (mL):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.internal_volume, width=10).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(meta, text="Solvent Name:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.solvent_name, width=16).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(meta, text="Product Name:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.product_name, width=16).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(meta, text="Product MW (g/mol):").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.product_mw, width=10).grid(row=1, column=3, sticky="w", **pad)
        ttk.Label(meta, text="Task1:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.task1, width=16).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(meta, text="Task2:").grid(row=2, column=2, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.task2, width=16).grid(row=2, column=3, sticky="w", **pad)
        ttk.Label(meta, text="Task3:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(meta, textvariable=self.task3, width=16).grid(row=3, column=1, sticky="w", **pad)

        task_frame = ttk.LabelFrame(self, text="Task-specific names / molecular weights (optional)")
        task_frame.pack(fill="x", padx=10, pady=6)

        ttk.Label(
            task_frame,
            text="Empty fields fall back to the global product/reactant values above.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=11, sticky="w", **pad)

        ttk.Label(task_frame, text="Task", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(task_frame, text="Product Name", font=("Segoe UI", 9, "bold")).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(task_frame, text="Product MW", font=("Segoe UI", 9, "bold")).grid(row=1, column=2, sticky="w", **pad)

        for t_idx in range(3):
            row = 2 + t_idx
            ttk.Label(task_frame, text=f"Task{t_idx + 1}").grid(row=row, column=0, sticky="w", **pad)
            ttk.Entry(task_frame, textvariable=self.task_product_name_vars[t_idx][0], width=18).grid(row=row, column=1, sticky="w", **pad)
            ttk.Entry(task_frame, textvariable=self.task_product_mw_vars[t_idx][0], width=10).grid(row=row, column=2, sticky="w", **pad)

        start_row = 6
        ttk.Label(task_frame, text="Task", font=("Segoe UI", 9, "bold")).grid(row=start_row, column=0, sticky="w", **pad)
        for i in range(5):
            ttk.Label(task_frame, text=f"R{i+1} Name", font=("Segoe UI", 9, "bold")).grid(row=start_row, column=1 + 2*i, sticky="w", **pad)
            ttk.Label(task_frame, text=f"R{i+1} MW", font=("Segoe UI", 9, "bold")).grid(row=start_row, column=2 + 2*i, sticky="w", **pad)

        for t_idx in range(3):
            row = start_row + 1 + t_idx
            ttk.Label(task_frame, text=f"Task{t_idx + 1}").grid(row=row, column=0, sticky="w", **pad)
            for i in range(5):
                ttk.Entry(task_frame, textvariable=self.task_reactant_name_vars[t_idx][i], width=14).grid(row=row, column=1 + 2*i, sticky="w", **pad)
                ttk.Entry(task_frame, textvariable=self.task_reactant_mw_vars[t_idx][i], width=8).grid(row=row, column=2 + 2*i, sticky="w", **pad)

        react = ttk.LabelFrame(self, text="Reactants (mass fraction bounds)")
        react.pack(fill="x", padx=10, pady=6)

        headers = ["#", "Name", "MW (g/mol)", "Limiting Flag", "Mode", "MF min", "MF max / fixed"]
        for c, h in enumerate(headers):
            ttk.Label(react, text=h, font=("Segoe UI", 9, "bold")).grid(row=0, column=c, sticky="w", **pad)

        for i in range(5):
            row = i + 1
            ttk.Label(react, text=str(i + 1)).grid(row=row, column=0, sticky="w", **pad)

            name = tk.StringVar(value=f"Reactant {i+1}")
            mw = tk.StringVar(value="")
            limiting = tk.IntVar(value=1 if i < 2 else 0)  # common default: 1&2 limiting candidates
            mode = tk.StringVar(value="range")
            mf_min = tk.StringVar(value="0.0")
            mf_max = tk.StringVar(value="1.0")

            ttk.Entry(react, textvariable=name, width=22).grid(row=row, column=1, sticky="w", **pad)
            ttk.Entry(react, textvariable=mw, width=10).grid(row=row, column=2, sticky="w", **pad)
            ttk.Checkbutton(react, variable=limiting).grid(row=row, column=3, sticky="w", **pad)

            cmb = ttk.Combobox(react, textvariable=mode, values=["range", "fixed"], width=8, state="readonly")
            cmb.grid(row=row, column=4, sticky="w", **pad)

            ttk.Entry(react, textvariable=mf_min, width=10).grid(row=row, column=5, sticky="w", **pad)
            ttk.Entry(react, textvariable=mf_max, width=14).grid(row=row, column=6, sticky="w", **pad)

            self.reactant_entries.append(
                {"name": name, "mw": mw, "limiting": limiting, "mode": mode, "mf_min": mf_min, "mf_max": mf_max}
            )


        proc = ttk.LabelFrame(self, text="Process bounds")
        proc.pack(fill="x", padx=10, pady=6)

        ttk.Label(proc, text="Temperature (°C) min/max:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.temp_min_var, width=8).grid(row=0, column=1, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.temp_max_var, width=8).grid(row=0, column=2, sticky="w", **pad)

        ttk.Label(proc, text="Time (min) min/max:").grid(row=0, column=3, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.time_min_var, width=8).grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.time_max_var, width=8).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(proc, text="Milling Load (mg/mL) min/max:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.milling_min_var, width=8).grid(row=1, column=1, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.milling_max_var, width=8).grid(row=1, column=2, sticky="w", **pad)

        ttk.Label(proc, text="Liquid Additive (uL/mg) min/max:").grid(row=1, column=3, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.addliq_min_var, width=8).grid(row=1, column=4, sticky="w", **pad)
        ttk.Entry(proc, textvariable=self.addliq_max_var, width=8).grid(row=1, column=5, sticky="w", **pad)

        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=10, pady=10)

        ttk.Button(bot, text="Create CSV…", command=self.create_csv_file).pack(side="left")
        ttk.Label(bot, text="Creates a two-block CSV compatible with the BO GUI.").pack(side="left", padx=10)

    def _toggle_reactant_rows(self) -> None:
        raw = (self.num_of_reactants.get() or "").strip()
        if raw == "":
            return
        n = _safe_int(raw, default=3)
        if n is None:
            n = 3
        n = int(min(max(n, 2), 5))
        # normalize the entry to the clamped integer
        self.num_of_reactants.set(str(n))

        # Just grey out unused reactant lines (name/mw/mf boxes)
        for i, e in enumerate(self.reactant_entries):
            state = "normal" if i < n else "disabled"
            # ttk.Entry widgets are not stored directly; easiest: set vars to blanks for disabled
            if state == "disabled":
                if e["mw"].get().strip() == "":
                    pass
                e["limiting"].set(0)
                e["mode"].set("range")
                e["mf_min"].set("0.0")
                e["mf_max"].set("0.0")

    def _collect_specs(self) -> Tuple[int, int, float, str, str, float, List[ReactantSpec]]:
        n = int(self.num_of_reactants.get())
        n_rows = _safe_int(self.n_initial_rows.get(), default=5) or 5
        n_rows = max(1, int(n_rows))

        internal_volume = _safe_float(self.internal_volume.get(), default=np.nan)
        if not np.isfinite(internal_volume) or internal_volume <= 0:
            raise ValueError("Internal Volume (mL) must be a positive number.")

        solvent = self.solvent_name.get().strip() or "xxx"
        prod_name = self.product_name.get().strip() or "product"
        prod_mw = _safe_float(self.product_mw.get(), default=np.nan)
        if not np.isfinite(prod_mw) or prod_mw <= 0:
            raise ValueError("Product MW (g/mol) must be a positive number.")

        specs: List[ReactantSpec] = []
        for i in range(5):
            e = self.reactant_entries[i]
            if i < n:
                name = e["name"].get().strip() or f"Reactant {i+1}"
                mw = _safe_float(e["mw"].get(), default=np.nan)
                if not np.isfinite(mw) or mw <= 0:
                    raise ValueError(f"Reactant {i+1}: MW (g/mol) must be a positive number.")

                mode = e["mode"].get()
                mf_min = _safe_float(e["mf_min"].get(), default=np.nan)
                mf_max = _safe_float(e["mf_max"].get(), default=np.nan)
                if mode == "fixed":
                    mf_min = mf_max  # fixed uses the max box
                if not (np.isfinite(mf_min) and np.isfinite(mf_max) and 0.0 <= mf_min <= mf_max <= 1.0):
                    raise ValueError(f"Reactant {i+1}: MassFrac bounds must be within [0,1] and min<=max.")
                limiting = int(e["limiting"].get())
                specs.append(ReactantSpec(name, float(mw), limiting, mode, float(mf_min), float(mf_max)))
            else:
                specs.append(ReactantSpec(f"Reactant {i+1}", np.nan, 0, "range", 0.0, 0.0))


        # Validate optional task-specific MW fields.
        for t_idx, task_var in enumerate([self.task1, self.task2, self.task3]):
            task_name = task_var.get().strip()
            if not task_name:
                continue

            task_prod_mw_raw = self.task_product_mw_vars[t_idx][0].get().strip()
            if task_prod_mw_raw:
                task_prod_mw = _safe_float(task_prod_mw_raw, default=np.nan)
                if not np.isfinite(task_prod_mw) or task_prod_mw <= 0:
                    raise ValueError(f"Task{t_idx + 1}: Product MW must be a positive number.")

            for i in range(n):
                mw_raw = self.task_reactant_mw_vars[t_idx][i].get().strip()
                if mw_raw:
                    mw_val = _safe_float(mw_raw, default=np.nan)
                    if not np.isfinite(mw_val) or mw_val <= 0:
                        raise ValueError(f"Task{t_idx + 1}, Reactant {i+1}: MW must be a positive number.")

        return n, n_rows, float(internal_volume), solvent, prod_name, float(prod_mw), specs

    def create_csv_file(self) -> None:
        try:
            n, n_rows, internal_volume, solvent, prod_name, prod_mw, specs = self._collect_specs()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Process bounds
        temp_min = _safe_int(self.temp_min_var.get(), default=25)
        temp_max = _safe_int(self.temp_max_var.get(), default=temp_min if temp_min is not None else 25)
        time_min = _safe_int(self.time_min_var.get(), default=5)
        time_max = _safe_int(self.time_max_var.get(), default=time_min if time_min is not None else 5)
        milling_min = _safe_float(self.milling_min_var.get(), default=1.0)
        milling_max = _safe_float(self.milling_max_var.get(), default=milling_min)
        addliq_min = _safe_float(self.addliq_min_var.get(), default=0.0)
        addliq_max = _safe_float(self.addliq_max_var.get(), default=addliq_min)

        if temp_min is None or temp_max is None or time_min is None or time_max is None:
            messagebox.showerror("Error", "Temperature and Time must be integers.")
            return
        if temp_min > temp_max or time_min > time_max:
            messagebox.showerror("Error", "Temperature/Time min must be <= max.")
            return
        if milling_min > milling_max or addliq_min > addliq_max:
            messagebox.showerror("Error", "Milling/Additive min must be <= max.")
            return

        # Save dialog
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        path = str(Path(path))

        # ----------------------------
        # Build metadata block
        # ----------------------------
        meta_rows = [
            ["Internal Volume (mL)", _fmt(internal_volume)],
            ["Solvent Name", solvent],
            ["Number of Reactants", str(n)],
            ["Product Name", prod_name],
            ["Product MW (g/mol)", _fmt(prod_mw)],
        ]
        for i in range(5):
            meta_rows += [
                [f"Reactant {i+1} Name", specs[i].name],
                [f"Reactant {i+1} MW (g/mol)", _fmt(specs[i].mw)],
                [f"Reactant {i+1} Limiting Flag", str(int(specs[i].limiting_flag))],
            ]
        meta_rows += [
            ["Task1", self.task1.get().strip()],
            ["Task2", self.task2.get().strip()],
            ["Task3", self.task3.get().strip()],
        ]

        # Task-specific product/reactant metadata. Empty GUI fields are written as global fallbacks.
        task_vars = [self.task1, self.task2, self.task3]
        for t_idx, task_var in enumerate(task_vars):
            task_label = task_var.get().strip()
            if not task_label:
                continue

            prod_name_task = self.task_product_name_vars[t_idx][0].get().strip() or prod_name
            prod_mw_task_raw = self.task_product_mw_vars[t_idx][0].get().strip()
            prod_mw_task = _safe_float(prod_mw_task_raw, default=prod_mw) if prod_mw_task_raw else prod_mw

            meta_rows += [
                [f"Task{t_idx + 1} Product Name", prod_name_task],
                [f"Task{t_idx + 1} Product MW (g/mol)", _fmt(prod_mw_task)],
            ]

            for i in range(5):
                r_name_task = self.task_reactant_name_vars[t_idx][i].get().strip() or specs[i].name
                r_mw_raw = self.task_reactant_mw_vars[t_idx][i].get().strip()
                r_mw_task = _safe_float(r_mw_raw, default=specs[i].mw) if r_mw_raw else specs[i].mw
                meta_rows += [
                    [f"Task{t_idx + 1} Reactant {i+1} Name", r_name_task],
                    [f"Task{t_idx + 1} Reactant {i+1} MW (g/mol)", _fmt(r_mw_task)],
                ]
        meta_rows.append(["", ""])  # optional visual spacer (still part of block 1)

        # ----------------------------
        # Build header (block 2)
        # ----------------------------
        header = [
            "metadata",
            "Task",
            "Reaction Number",
            "Iteration Number",
            "n moles",
        ]
        for i in range(5):
            header += [f"Reactant {i+1} Mass", f"Reactant {i+1} MassFrac"]
        header += [
            "Temperature (°C)",
            "Reaction Time (min)",
            "Milling Load (mg/mL)",
            "Liquid Additive (uL/mg)",
            "NMR_conversion",
            "Productivity",
            "PMI",
            "target",
            "mass_of_product",
        ]
        for i in range(5):
            header += [
                f"Reactant {i+1} MassFrac Range Bottom",
                f"Reactant {i+1} MassFrac Range Top",
            ]
        header += [
            "Temperature Min (°C)",
            "Temperature Max (°C)",
            "Time Min (min)",
            "Time Max (min)",
            "Milling Load Min (mg/mL)",
            "Milling Load Max (mg/mL)",
            "Liquid Additive Min (uL/mg)",
            "Liquid Additive Max (uL/mg)",
        ]

        # Constant bounds row values
        mf_bottom = [specs[i].mf_min if i < n else np.nan for i in range(5)]
        mf_top = [specs[i].mf_max if i < n else np.nan for i in range(5)]

        # Sample initial rows
        mins = np.array([specs[i].mf_min for i in range(n)], dtype=float)
        maxs = np.array([specs[i].mf_max for i in range(n)], dtype=float)
        modes = [specs[i].mode for i in range(n)]

        data_rows: List[List[str]] = []
        for k in range(n_rows):
            # process vars
            temp = random.randint(int(temp_min), int(temp_max)) if temp_min != temp_max else int(temp_min)
            time = random.randint(int(time_min), int(time_max)) if time_min != time_max else int(time_min)
            milling = random.uniform(float(milling_min), float(milling_max)) if milling_min != milling_max else float(milling_min)
            addliq = random.uniform(float(addliq_min), float(addliq_max)) if addliq_min != addliq_max else float(addliq_min)

            try:
                mf_active = _sample_massfracs_with_bounds(mins, maxs, modes)
            except Exception as e:
                messagebox.showerror("Error", f"Could not sample mass fractions: {e}")
                return

            util_mass = internal_volume * milling  # mg (if internal_volume mL and milling mg/mL)
            masses_active = util_mass * mf_active  # mg

            # Expand to 5 reactants
            mf_full = [np.nan] * 5
            mass_full = [np.nan] * 5
            for i in range(n):
                mf_full[i] = float(mf_active[i])
                mass_full[i] = float(masses_active[i])

            row = [""] * len(header)
            # metadata col stays blank per your examples
            def setcol(colname: str, value):
                j = header.index(colname)
                row[j] = _fmt(value)

            setcol("Task", self.task1.get().strip())
            setcol("Reaction Number", k + 1)
            setcol("Iteration Number", 1)
            setcol("n moles", np.nan)

            for i in range(5):
                setcol(f"Reactant {i+1} Mass", mass_full[i])
                setcol(f"Reactant {i+1} MassFrac", mf_full[i])

            setcol("Temperature (°C)", temp)
            setcol("Reaction Time (min)", time)
            setcol("Milling Load (mg/mL)", milling)
            setcol("Liquid Additive (uL/mg)", addliq)

            # Results blank
            setcol("NMR_conversion", np.nan)
            setcol("Productivity", np.nan)
            setcol("PMI", np.nan)
            setcol("target", np.nan)
            setcol("mass_of_product", np.nan)

            # Bounds
            for i in range(5):
                setcol(f"Reactant {i+1} MassFrac Range Bottom", mf_bottom[i])
                setcol(f"Reactant {i+1} MassFrac Range Top", mf_top[i])

            setcol("Temperature Min (°C)", temp_min)
            setcol("Temperature Max (°C)", temp_max)
            setcol("Time Min (min)", time_min)
            setcol("Time Max (min)", time_max)
            setcol("Milling Load Min (mg/mL)", milling_min)
            setcol("Milling Load Max (mg/mL)", milling_max)
            setcol("Liquid Additive Min (uL/mg)", addliq_min)
            setcol("Liquid Additive Max (uL/mg)", addliq_max)

            data_rows.append(row)

        # ----------------------------
        # Write two-block CSV
        # ----------------------------
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for r in meta_rows:
                    w.writerow(r)
                w.writerow([])  # separator row
                w.writerow(header)
                for r in data_rows:
                    w.writerow(r)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write CSV:\n{e}")
            return

        messagebox.showinfo("Done", f"CSV created:\n{path}")


def main() -> None:
    app = InitializationMassFracGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
