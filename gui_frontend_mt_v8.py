# gui_frontend_mt_v3.py
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import re
from datetime import datetime


def read_two_block_csv(path_or_text):
    """
    Read the two-block CSV format:
      metadata key,value rows
      blank separator
      table header + rows
    Accepts a file path or raw CSV text.
    """
    import io

    if isinstance(path_or_text, str) and ("\n" in path_or_text):
        text = path_or_text
    else:
        with open(path_or_text, "r", encoding="utf-8") as f:
            text = f.read()

    lines = text.splitlines()
    sep_idx = None
    for i, line in enumerate(lines):
        if not line.strip():
            sep_idx = i
            break
    if sep_idx is None:
        raise ValueError("Could not find blank separator row between metadata and table.")

    metadata = {}
    for line in lines[:sep_idx]:
        if not line.strip():
            continue
        parts = line.split(",", 1)
        key = parts[0].strip()
        val = parts[1].strip() if len(parts) > 1 else ""
        if key:
            metadata[key] = val

    table_text = "\n".join(lines[sep_idx + 1:])
    if not table_text.strip():
        raise ValueError("CSV table block is empty.")
    df = pd.read_csv(io.StringIO(table_text))
    return metadata, df


from bo_data_mt_v3 import (
    parse_reactants_from_metadata,
    infer_bounds_from_last_row,
    build_XY_from_df,
    build_multitask_design,
    expand_simplex_X,
    get_task_chemistry,
)
from bo_engine_botorch_mt_v6 import (
    fit_single_task_gp,
    suggest_next_ei,
    suggest_q_logei,
    fit_multitask_gp,
    best_f_for_task,
    suggest_next_mt_ei,
    suggest_q_logei_mt,
)
from bo_metrics_mt_v4 import refresh_table


def write_two_block_csv(path: str, metadata: dict, df: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metadata.items():
            f.write(f"{k},{v}\n")
        f.write("\n")
        df.to_csv(f, index=False)


class BayesianOptimizationGUI:
    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _get_decimals_var_for_col(self, col: str) -> tk.IntVar:
        """Get (or create) the decimals IntVar for a given column."""
        col = str(col)
        v = self.col_decimals.get(col)
        if v is None:
            # default rules (can be overridden in Options)
            d = int(self.default_decimals_var.get())
            c = col.lower()
            if "massfrac" in c:
                d = 4
            elif "temperature" in c or "time (min)" in c or "reaction time" in c:
                d = 0
            elif "equiv" in c:
                d = 3
            elif "mass" in c or "volume" in c or "milling" in c or "additive" in c:
                d = 3
            v = tk.IntVar(value=max(0, min(10, d)))
            self.col_decimals[col] = v
        return v

    def fmt_val(self, v, col: str | None = None) -> str:
        """Format values for display without altering stored data."""
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass

        # strings: try numeric, otherwise return as-is
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                return ""
            try:
                x = float(s)
            except Exception:
                return v
        else:
            try:
                x = float(v)
            except Exception:
                return str(v)

        if float(x).is_integer():
            return str(int(x))

        if col is None:
            d = int(self.default_decimals_var.get())
        else:
            d = int(self._get_decimals_var_for_col(col).get())
        d = max(0, min(12, d))
        return f"{x:.{d}f}"

    def _refresh_all_displays(self) -> None:
        """Refresh Treeview + To-do cards (used after option changes)."""
        self._refresh_tree()
        self.refresh_todo_list()

    def _update_task_choices_from_metadata(self) -> None:
        """Read possible task labels from metadata keys Task1, Task2, Task3, ..."""
        self._task_choices = []
        if self.metadata:
            for i in range(1, 21):
                key = f"Task{i}"
                val = str(self.metadata.get(key, "") or "").strip()
                if val:
                    self._task_choices.append(val)

        if hasattr(self, "task_combo"):
            try:
                self.task_combo["values"] = self._task_choices
            except Exception:
                pass

        # Pick a sensible default if current task is empty
        cur = (self.active_task_var.get() or "").strip()
        if not cur:
            if self._task_choices:
                self.active_task_var.set(self._task_choices[0])
            elif self.df is not None and "Task" in self.df.columns and len(self.df) > 0:
                vals = self.df["Task"].dropna().astype(str).str.strip()
                vals = vals[vals != ""]
                if not vals.empty:
                    self.active_task_var.set(vals.iloc[-1])



    def _simplex_full_bounds_from_last_row(self, n_reactants: int) -> np.ndarray | None:
        """Return full Reactant 1..N MassFrac bounds as shape (2,N)."""
        if self.df is None or len(self.df) == 0 or int(n_reactants) <= 1:
            return None
        last = self.df.iloc[-1]
        lows = []
        highs = []
        for i in range(1, int(n_reactants) + 1):
            lo_col = f"Reactant {i} MassFrac Range Bottom"
            hi_col = f"Reactant {i} MassFrac Range Top"
            if lo_col not in self.df.columns or hi_col not in self.df.columns:
                return None
            lows.append(float(last[lo_col]))
            highs.append(float(last[hi_col]))
        return np.array([lows, highs], dtype=float)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bayesian Optimization GUI (Multitask v8)")
        self.root.geometry("1300x750")

        self.file_path: str | None = None
        self.metadata: dict[str, str] | None = None
        self.df: pd.DataFrame | None = None

        self.objective_var = tk.StringVar(value="target")
        # Treeview zoom (affects font size + row height + column width)
        self.zoom_var = tk.IntVar(value=100)
        # Number formatting
        self.default_decimals_var = tk.IntVar(value=3)
        self.col_decimals: dict[str, tk.IntVar] = {}
        self.batch_q_var = tk.StringVar(value="1")
        self.active_task_var = tk.StringVar(value="")
        self._task_choices: list[str] = []
        self._nmr_entries: dict[int, tk.StringVar] = {}
        # Column visibility state (populated after loading a CSV)
        self.col_visible: dict[str, tk.BooleanVar] = {}
        self._options_win: tk.Toplevel | None = None

        self.style = ttk.Style(self.root)
        self._build_ui()
        self._apply_tree_zoom()

    def _resize_tree_columns(self) -> None:
        """Resize Treeview columns based on current heading font + heading text."""
        if not hasattr(self, "tree"):
            return
        cols = list(self.tree["columns"]) if self.tree is not None else []
        if not cols:
            return
        heading_font = getattr(self, "_heading_font", tkfont.Font(family="Segoe UI", size=10, weight="bold"))
        pad = int(12 + heading_font.cget("size") * 1.8)
        min_w = 80
        for c in cols:
            title = self.tree.heading(c, option="text") or str(c)
            w = max(min_w, heading_font.measure(title) + pad)
            try:
                self.tree.column(c, width=w, minwidth=min_w, stretch=False)
            except Exception:
                pass

    def _apply_tree_zoom(self) -> None:
        """Apply zoom settings to Treeview: font size + row height, and keep a matching heading font for sizing."""
        z = max(60, min(160, int(self.zoom_var.get())))
        font_size = max(8, round(10 * z / 100))
        row_h = max(14, round(22 * z / 100))
        self._heading_font = tkfont.Font(family="Segoe UI", size=font_size, weight="bold")
        self.style.configure("Treeview", font=("Segoe UI", font_size), rowheight=row_h)
        self.style.configure("Treeview.Heading", font=("Segoe UI", font_size, "bold"))
        self._resize_tree_columns()

    def _init_column_visibility(self) -> None:
        """Initialize column visibility for columns present in df.
        Keeps existing user choices and only adds toggles for new columns.
        """
        if self.df is None:
            return

        for c in self.df.columns:
            cs = str(c)
            if cs in self.col_visible:
                continue  # keep user's existing preference

            # default rules (can be overridden in Options)
            default_on = True
            if any(p in cs for p in ("Range", " Min", " Max")):
                default_on = False
            if " Moles" in cs:
                default_on = False
            if cs in ("created_at", "completed_at"):
                default_on = False
            if cs.endswith("_proposed"):
                default_on = False
            if cs.startswith("Liquid Additive Volume"):
                default_on = False
            if cs == "Notes":
                default_on = False
            ser = self.df[c].replace("", np.nan)
            if ser.isna().all():
                default_on = False

            self.col_visible[cs] = tk.BooleanVar(value=default_on)
            self._get_decimals_var_for_col(cs)

    def open_options(self) -> None:
        if self.df is None:
            messagebox.showinfo("Options", "Load a CSV first.")
            return
        if self._options_win is not None and self._options_win.winfo_exists():
            self._options_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Options")
        win.geometry("420x600")
        self._options_win = win

        ttk.Label(win, text="Show/Hide columns", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 6))

        # Zoom control (font + row height)
        zfr = ttk.Frame(win)
        zfr.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(zfr, text="Zoom (table):").pack(side="left")
        zscale = ttk.Scale(zfr, from_=60, to=160, orient="horizontal", variable=self.zoom_var,
                           command=lambda _=None: (self._apply_tree_zoom(), self._refresh_tree()))
        zscale.pack(side="left", fill="x", expand=True, padx=8)
        zval = ttk.Label(zfr, textvariable=self.zoom_var, width=4)
        zval.pack(side="right")


        # Scrollable list of checkboxes
        container = ttk.Frame(win)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        canvas = tk.Canvas(container, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scroll.set)

        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=frame, anchor="nw")

        # Bulk actions
        bulk = ttk.Frame(win)
        bulk.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bulk, text="Show all", command=lambda: self._set_all_columns(True)).pack(side="left")
        ttk.Button(bulk, text="Hide all", command=lambda: self._set_all_columns(False)).pack(side="left", padx=8)
        ttk.Button(bulk, text="Apply", command=self._apply_column_visibility).pack(side="right")

        # Create per-column controls (visibility + decimals)
        header_row = ttk.Frame(frame)
        header_row.pack(fill="x", pady=(0, 6))
        ttk.Label(header_row, text="Column", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header_row, text="Decimals", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(12, 0))

        for c in map(str, self.df.columns):
            var = self.col_visible.get(c)
            if var is None:
                var = tk.BooleanVar(value=True)
                self.col_visible[c] = var

            dvar = self._get_decimals_var_for_col(c)

            rowf = ttk.Frame(frame)
            rowf.pack(fill="x", pady=2)

            ttk.Checkbutton(rowf, text=c, variable=var).grid(row=0, column=0, sticky="w")
            sp = ttk.Spinbox(rowf, from_=0, to=10, width=4, textvariable=dvar)
            sp.grid(row=0, column=1, sticky="e", padx=(12, 0))
            sp.bind("<Return>", lambda e: self._refresh_all_displays())
            sp.bind("<FocusOut>", lambda e: self._refresh_all_displays())


        win.protocol("WM_DELETE_WINDOW", win.destroy)

    def _set_all_columns(self, state: bool) -> None:
        for var in self.col_visible.values():
            var.set(bool(state))

    def _apply_column_visibility(self) -> None:
        self._refresh_tree()

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(expand=True, fill="both")
        self.notebook = nb

        main = ttk.Frame(nb)
        todo = ttk.Frame(nb)
        nb.add(main, text="Main")
        nb.add(todo, text="To do list")

        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        toolbar = ttk.Frame(main)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        toolbar.grid_columnconfigure(10, weight=1)

        ttk.Button(toolbar, text="Open CSV", command=self.open_csv).grid(row=0, column=0, padx=4)
        ttk.Button(toolbar, text="Save CSV", command=self.save_csv).grid(row=0, column=1, padx=4)
        ttk.Label(toolbar, text="Objective:").grid(row=0, column=2, padx=(16, 4))
        ttk.Entry(toolbar, textvariable=self.objective_var, width=18).grid(row=0, column=3, padx=4)
        ttk.Label(toolbar, text="Batch q:").grid(row=0, column=4, padx=(16, 4))
        ttk.Entry(toolbar, textvariable=self.batch_q_var, width=4).grid(row=0, column=5, padx=(0, 4))
        ttk.Label(toolbar, text="Task:").grid(row=0, column=6, padx=(16, 4))
        self.task_combo = ttk.Combobox(toolbar, textvariable=self.active_task_var, values=self._task_choices, width=18)
        self.task_combo.grid(row=0, column=7, padx=(0, 4))
        ttk.Button(toolbar, text="Suggest next (BoTorch)", command=self.suggest_next).grid(row=0, column=8, padx=(8, 4))

        ttk.Button(toolbar, text="Options…", command=self.open_options).grid(row=0, column=9, padx=(16, 4))


        self.file_label = ttk.Label(toolbar, text="No file loaded")
        self.file_label.grid(row=0, column=10, sticky="w")

        table_frame = ttk.Frame(main)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # todo
        top = ttk.Frame(todo)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Refresh list", command=self.refresh_todo_list).pack(side="left")
        ttk.Button(top, text="Update NMR values", command=self.update_nmr_values).pack(side="left", padx=8)

        container = ttk.Frame(todo)
        container.pack(expand=True, fill="both", padx=8, pady=(0, 8))

        canvas = tk.Canvas(container, highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scroll.set)

        self.todo_frame = ttk.Frame(canvas)
        self.todo_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.todo_frame, anchor="nw")

    def _refresh_tree(self) -> None:
        if self.df is None:
            return

        self.tree.delete(*self.tree.get_children())

        # Column visibility is controlled by Options window (self.col_visible).
        cols_all = list(self.df.columns)
        visible = []
        for c in cols_all:
            cs = str(c)
            var = self.col_visible.get(cs)
            if var is None:
                # if not initialized yet, default to True
                var = tk.BooleanVar(value=True)
                self.col_visible[cs] = var
            if not bool(var.get()):
                continue
            visible.append(c)

        # Second pass: build preferred order
        ordered: list[str] = []

        def _add(col: str) -> None:
            if col in visible and col not in ordered:
                ordered.append(col)

        # IDs first
        _add("Reaction Number")
        _add("Iteration Number")

        # Reactant blocks: Equiv i, MassFrac i, Mass i
        max_i = 0
        for c in visible:
            m = re.match(r"Reactant (\d+)\s", str(c))
            if m:
                max_i = max(max_i, int(m.group(1)))

        for i in range(1, max_i + 1):
            _add(f"Reactant {i} Equiv")
            _add(f"Reactant {i} MassFrac")
            _add(f"Reactant {i} Mass")

        # Process conditions
        _add("Temperature (°C)")
        _add("Reaction Time (min)")
        _add("Milling Load (mg/mL)")
        _add("Liquid Additive (uL/mg)")

        # Results / metrics
        _add("NMR_conversion")
        _add("mass_of_product")
        _add("PMI")
        _add("Productivity")
        _add("target")
        _add("n moles")

        # Append remaining visible columns (stable order)
        for c in visible:
            if c not in ordered:
                ordered.append(c)

        self.tree["columns"] = ordered
        self.tree["show"] = "headings"

        self._apply_tree_zoom()
        heading_font = getattr(self, '_heading_font', tkfont.Font(family='Segoe UI', size=10, weight='bold'))
        pad = int(12 + heading_font.cget('size') * 1.8)  # pixels; scales with font size
        min_w = 80

        for c in ordered:
            title = str(c)
            self.tree.heading(c, text=title)
            w = max(min_w, heading_font.measure(title) + pad)
            self.tree.column(c, width=w, minwidth=min_w, stretch=False)

        self._apply_tree_zoom()

        for _, row in self.df.iterrows():
            self.tree.insert("", "end", values=[self.fmt_val(row.get(c, ""), c) for c in ordered])

    def refresh_todo_list(self) -> None:
        for w in self.todo_frame.winfo_children():
            w.destroy()

        if self.df is None or self.metadata is None:
            ttk.Label(self.todo_frame, text="Load a CSV first.").pack(anchor="w")
            return

        nmr_col = "NMR_conversion" if "NMR_conversion" in self.df.columns else None
        if nmr_col is None:
            ttk.Label(self.todo_frame, text="No NMR_conversion column found.").pack(anchor="w")
            return

        # Determine pending rows (no NMR)
        missing = pd.to_numeric(self.df[nmr_col], errors="coerce").isna()
        idxs = list(self.df.index[missing])
        if not idxs:
            ttk.Label(self.todo_frame, text="No rows missing NMR_conversion.").pack(anchor="w")
            return

        try:
            _, _, _, n_reactants, reactants = parse_reactants_from_metadata(self.metadata)
            n_reactants = int(n_reactants)
            names = [r.name for r in reactants]
        except Exception:
            names = []
            n_reactants = 0

        # Ensure columns exist
        if "Notes" not in self.df.columns:
            self.df["Notes"] = pd.Series(dtype="object")
        if "Liquid Additive Volume (uL)" not in self.df.columns:
            self.df["Liquid Additive Volume (uL)"] = np.nan

        # Compute additive volume from ratio if possible (for display)
        if "Liquid Additive (uL/mg)" in self.df.columns and "util_mass" in self.df.columns:
            self.df["Liquid Additive Volume (uL)"] = pd.to_numeric(self.df["Liquid Additive (uL/mg)"], errors="coerce") * pd.to_numeric(self.df["util_mass"], errors="coerce")

        for idx in idxs:
            row = self.df.loc[idx]
            rxn_num = row.get("Reaction Number", idx + 1)

            # Use task-specific reactant names when available.
            task_names = names
            if self.metadata is not None and "Task" in self.df.columns:
                task_label = str(row.get("Task", "") or "").strip()
                try:
                    chem = get_task_chemistry(self.metadata, task_label, allow_global_fallback=True)
                    task_names = [r.name for r in chem.reactants]
                except Exception:
                    task_names = names

            lf = ttk.LabelFrame(self.todo_frame, text=f"Reaction {rxn_num}")
            lf.pack(fill="x", padx=6, pady=6)

            # Build grid: Proposed vs Actual
            ttk.Label(lf, text="Proposed", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w", padx=6, pady=(6, 2))
            ttk.Label(lf, text="Actual", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))

            r = 1
            mass_vars: dict[str, tk.StringVar] = {}

            for i in range(1, n_reactants + 1):
                name = f"Reactant {i} Mass"
                prop = f"Reactant {i} Mass_proposed"
                reactant_label = task_names[i-1] if i-1 < len(task_names) else f"Reactant {i}"
                ttk.Label(lf, text=f"{reactant_label} mass").grid(row=r, column=0, sticky="w", padx=6, pady=2)
                prop_val = row.get(prop, row.get(name, ""))
                ttk.Label(lf, text=self.fmt_val(prop_val, prop)).grid(row=r, column=1, sticky="w", padx=6, pady=2)
                mass_val = row.get(name, np.nan)
                v = tk.StringVar(value="" if pd.isna(mass_val) else self.fmt_val(mass_val, name))
                ttk.Entry(lf, textvariable=v, width=14).grid(row=r, column=2, sticky="w", padx=6, pady=2)
                ttk.Label(lf, text="mg").grid(row=r, column=3, sticky="w", padx=(0, 6))
                mass_vars[name] = v
                r += 1
            # Temperature
            temp_name = "Temperature (°C)"
            temp_prop = "Temperature (°C)_proposed"
            ttk.Label(lf, text="Temperature").grid(row=r, column=0, sticky="w", padx=6, pady=2)
            ttk.Label(lf, text=self.fmt_val(row.get(temp_prop, row.get(temp_name, "")), temp_prop if temp_prop in row else temp_name)).grid(row=r, column=1, sticky="w", padx=6, pady=2)
            temp_val = row.get(temp_name, np.nan)
            temp_var = tk.StringVar(value="" if pd.isna(temp_val) else self.fmt_val(temp_val, temp_name))
            ttk.Entry(lf, textvariable=temp_var, width=14).grid(row=r, column=2, sticky="w", padx=6, pady=2)
            ttk.Label(lf, text="°C").grid(row=r, column=3, sticky="w", padx=(0, 6))
            r += 1

            # Time
            time_name = "Reaction Time (min)"
            time_prop = "Reaction Time (min)_proposed"
            ttk.Label(lf, text="Time").grid(row=r, column=0, sticky="w", padx=6, pady=2)
            ttk.Label(lf, text=self.fmt_val(row.get(time_prop, row.get(time_name, "")), time_prop if time_prop in row else time_name)).grid(row=r, column=1, sticky="w", padx=6, pady=2)
            time_val = row.get(time_name, np.nan)
            time_var = tk.StringVar(value="" if pd.isna(time_val) else self.fmt_val(time_val, time_name))
            ttk.Entry(lf, textvariable=time_var, width=14).grid(row=r, column=2, sticky="w", padx=6, pady=2)
            ttk.Label(lf, text="min").grid(row=r, column=3, sticky="w", padx=(0, 6))
            r += 1

            # Liquid additive volume
            addvol_name = "Liquid Additive Volume (uL)"
            addvol_prop = "Liquid Additive Volume (uL)_proposed"
            ttk.Label(lf, text="Liquid additive volume").grid(row=r, column=0, sticky="w", padx=6, pady=2)
            ttk.Label(lf, text=self.fmt_val(row.get(addvol_prop, row.get(addvol_name, "")), addvol_prop if addvol_prop in row else addvol_name)).grid(row=r, column=1, sticky="w", padx=6, pady=2)
            addvol_val = row.get(addvol_name, np.nan)
            addvol_var = tk.StringVar(value="" if pd.isna(addvol_val) else self.fmt_val(addvol_val, addvol_name))
            ttk.Entry(lf, textvariable=addvol_var, width=14).grid(row=r, column=2, sticky="w", padx=6, pady=2)
            ttk.Label(lf, text="uL").grid(row=r, column=3, sticky="w", padx=(0, 6))
            r += 1

            # NMR conversion
            ttk.Label(lf, text="NMR_conversion").grid(row=r, column=0, sticky="w", padx=6, pady=(6, 2))
            nmr_var = tk.StringVar(value="")
            ttk.Entry(lf, textvariable=nmr_var, width=14).grid(row=r, column=2, sticky="w", padx=6, pady=(6, 2))
            ttk.Label(lf, text="(0-100)").grid(row=r, column=3, sticky="w", padx=(0, 6), pady=(6, 2))
            r += 1

            # Notes
            ttk.Label(lf, text="Notes").grid(row=r, column=0, sticky="nw", padx=6, pady=(6, 2))
            notes = tk.Text(lf, height=3, width=50)
            existing_notes = row.get("Notes", "")
            if isinstance(existing_notes, str):
                notes.insert("1.0", existing_notes)
            notes.grid(row=r, column=1, columnspan=3, sticky="we", padx=6, pady=(6, 2))
            r += 1

            # Buttons
            btns = ttk.Frame(lf)
            btns.grid(row=r, column=0, columnspan=4, sticky="we", padx=6, pady=(6, 8))
            ttk.Button(
                btns,
                text="Save",
                command=lambda i=idx, mv=mass_vars, tv=temp_var, timv=time_var, av=addvol_var, nv=nmr_var, nt=notes: self._save_todo_row(
                    i, mv, tv, timv, av, nv, nt, mark_done=False
                ),
            ).pack(side="left")
            ttk.Button(
                btns,
                text="Mark completed",
                command=lambda i=idx, mv=mass_vars, tv=temp_var, timv=time_var, av=addvol_var, nv=nmr_var, nt=notes: self._save_todo_row(
                    i, mv, tv, timv, av, nv, nt, mark_done=True
                ),
            ).pack(side="left", padx=8)
            ttk.Button(btns, text="Reset actual → proposed", command=lambda i=idx: self._reset_actual_to_proposed(i)).pack(side="right")

            for c in range(4):
                lf.grid_columnconfigure(c, weight=1 if c == 2 else 0)

    def _reset_actual_to_proposed(self, idx: int) -> None:
        if self.df is None:
            return
        row = self.df.loc[idx]
        # copy proposed back to actual for the user-facing fields
        for c in list(self.df.columns):
            cs = str(c)
            if cs.endswith("_proposed"):
                base = cs.replace("_proposed", "")
                if base in self.df.columns:
                    self.df.at[idx, base] = row.get(c)

        # refresh computed volume too
        if "Liquid Additive (uL/mg)" in self.df.columns and "util_mass" in self.df.columns:
            self.df.at[idx, "Liquid Additive Volume (uL)"] = pd.to_numeric(self.df.at[idx, "Liquid Additive (uL/mg)"], errors="coerce") * pd.to_numeric(
                self.df.at[idx, "util_mass"], errors="coerce"
            )
        self._refresh_tree()
        self.refresh_todo_list()

    def _save_todo_row(
        self,
        idx: int,
        mass_vars: dict[str, tk.StringVar],
        temp_var: tk.StringVar,
        time_var: tk.StringVar,
        addvol_var: tk.StringVar,
        nmr_var: tk.StringVar,
        notes_widget: tk.Text,
        *,
        mark_done: bool,
    ) -> None:
        if self.df is None or self.metadata is None:
            return

        # Parse numeric inputs
        try:
            internal_vol, _, _, n_reactants, _ = parse_reactants_from_metadata(self.metadata)
            n_reactants = int(n_reactants)
            internal_vol = float(internal_vol)
        except Exception:
            n_reactants = 0
            internal_vol = float("nan")

        masses = []
        for i in range(1, n_reactants + 1):
            key = f"Reactant {i} Mass"
            v = mass_vars.get(key)
            val = float(v.get()) if v and v.get().strip() else float("nan")
            self.df.at[idx, key] = val
            masses.append(val)

        # Temp/time
        if "Temperature (°C)" in self.df.columns and temp_var.get().strip():
            self.df.at[idx, "Temperature (°C)"] = float(temp_var.get())
        if "Reaction Time (min)" in self.df.columns and time_var.get().strip():
            self.df.at[idx, "Reaction Time (min)"] = float(time_var.get())

        # util_mass from masses
        util_mass = float(np.nansum(masses)) if masses else float("nan")

        # Update mass fractions + milling load behind the scenes
        if np.isfinite(util_mass) and util_mass > 0:
            for i in range(1, n_reactants + 1):
                mcol = f"Reactant {i} Mass"
                mfcol = f"Reactant {i} MassFrac"
                if mfcol in self.df.columns:
                    mval = pd.to_numeric(self.df.at[idx, mcol], errors="coerce")
                    self.df.at[idx, mfcol] = float(mval) / util_mass if np.isfinite(mval) else np.nan

            # milling load (mg/mL)
            if "Milling Load (mg/mL)" in self.df.columns and np.isfinite(internal_vol) and internal_vol > 0:
                self.df.at[idx, "Milling Load (mg/mL)"] = util_mass / internal_vol

        # Liquid additive volume -> ratio (uL/mg)
        if "Liquid Additive Volume (uL)" not in self.df.columns:
            self.df["Liquid Additive Volume (uL)"] = np.nan
        addvol = float(addvol_var.get()) if addvol_var.get().strip() else float("nan")
        self.df.at[idx, "Liquid Additive Volume (uL)"] = addvol
        if "Liquid Additive (uL/mg)" in self.df.columns and np.isfinite(addvol) and np.isfinite(util_mass) and util_mass > 0:
            self.df.at[idx, "Liquid Additive (uL/mg)"] = addvol / util_mass

        # Notes
        notes = notes_widget.get("1.0", "end").strip()
        if "Notes" not in self.df.columns:
            self.df["Notes"] = pd.Series(dtype="object")
        self.df.at[idx, "Notes"] = notes

        # NMR conversion (optional)
        if nmr_var.get().strip():
            self.df.at[idx, "NMR_conversion"] = float(nmr_var.get())
            if "completed_at" in self.df.columns:
                self.df.at[idx, "completed_at"] = self._now_iso()

        if mark_done and "completed_at" in self.df.columns:
            self.df.at[idx, "completed_at"] = self._now_iso()

        # Recompute derived columns
        self.df = refresh_table(self.metadata, self.df)

        # Recompute additive volume for display
        if "Liquid Additive (uL/mg)" in self.df.columns and "util_mass" in self.df.columns:
            self.df["Liquid Additive Volume (uL)"] = pd.to_numeric(self.df["Liquid Additive (uL/mg)"], errors="coerce") * pd.to_numeric(self.df["util_mass"], errors="coerce")

        # Save back to same file if loaded from disk
        if self.file_path:
            try:
                write_two_block_csv(self.file_path, self.metadata, self.df)
            except Exception as e:
                messagebox.showerror("Save failed", f"Could not write CSV:\n{e}")
                return

        self._init_column_visibility()
        self._refresh_tree()
        self.refresh_todo_list()

    def open_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        try:
            meta, df = read_two_block_csv(path)
            df = refresh_table(meta, df)  # compute equivalents/PMI etc. using task-specific MWs when available
            if "Task" in df.columns:
                df["Task"] = df["Task"].astype("object")
            # Ensure timestamp columns exist (hidden in Treeview)
            if "created_at" not in df.columns:
                df["created_at"] = pd.Series(dtype="object")
            else:
                df["created_at"] = df["created_at"].astype("object")
            if "completed_at" not in df.columns:
                df["completed_at"] = pd.Series(dtype="object")
            else:
                df["completed_at"] = df["completed_at"].astype("object")
            # Ensure Notes column exists
            if "Notes" not in df.columns:
                df["Notes"] = pd.Series(dtype="object")
            else:
                df["Notes"] = df["Notes"].astype("object")

            # Liquid additive volume column (uL) for user-facing entry
            if "Liquid Additive Volume (uL)" not in df.columns:
                df["Liquid Additive Volume (uL)"] = np.nan

            # Ensure proposed columns exist (for traceability)
            try:
                internal_vol, _, _, n_reactants, _ = parse_reactants_from_metadata(meta)
                n_reactants = int(n_reactants)
            except Exception:
                n_reactants = 0
            for i in range(1, n_reactants + 1):
                colp = f"Reactant {i} Mass_proposed"
                if colp not in df.columns:
                    df[colp] = np.nan
            for colp in ("Temperature (°C)_proposed", "Reaction Time (min)_proposed", "Liquid Additive Volume (uL)_proposed"):
                if colp not in df.columns:
                    df[colp] = np.nan

            # Compute additive volume from ratio if possible
            if "Liquid Additive (uL/mg)" in df.columns and "util_mass" in df.columns:
                df["Liquid Additive Volume (uL)"] = pd.to_numeric(df["Liquid Additive (uL/mg)"], errors="coerce") * pd.to_numeric(df["util_mass"], errors="coerce")

            # Update the file on open so derived columns stay in sync
            write_two_block_csv(path, meta, df)
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not read CSV:\n{e}")
            return

        self.file_path = path
        self.metadata = meta
        self.df = df
        self._init_column_visibility()
        self._update_task_choices_from_metadata()
        self.file_label.config(text=f"Loaded: {path}")
        self._refresh_tree()
        self.refresh_todo_list()

    def save_csv(self) -> None:
        if self.df is None or self.metadata is None:
            messagebox.showwarning("No data", "Load a file first.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        try:
            write_two_block_csv(path, self.metadata, self.df)
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save:\n{e}")
            return

        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    def suggest_next(self) -> None:
        if self.df is None or self.metadata is None:
            messagebox.showwarning("No data", "Load a file first.")
            return

        objective = self.objective_var.get().strip()
        if not objective:
            messagebox.showwarning("Objective missing", "Enter an objective column name (e.g., target).")
            return
        if objective not in self.df.columns:
            messagebox.showerror("Objective missing", f"Column not found: {objective}")
            return

        try:
            _, _, _, n_reactants, _ = parse_reactants_from_metadata(self.metadata)
            simplex_bounds_full = self._simplex_full_bounds_from_last_row(int(n_reactants))

            # Batch size (q): editable field in the toolbar (defaults to 1)
            try:
                q = int(self.batch_q_var.get().strip() or "1")
            except ValueError:
                q = 1
            q = max(1, q)

            active_task = (self.active_task_var.get() or "").strip()
            use_mt = ("Task" in self.df.columns) and (active_task != "")

            if use_mt:
                # --- true multitask path ---
                design = build_multitask_design(
                    self.df,
                    metadata=self.metadata,
                    y_cols=[objective],
                    task_col="Task",
                    include_process=True,
                    dropna=True,
                    debug=False,
                )

                if design.X_aug.shape[0] < 3:
                    raise ValueError("Need at least 3 completed rows with objective values to fit a multitask GP.")

                if active_task not in design.task_to_index:
                    raise ValueError(
                        f"Selected task '{active_task}' was not found among completed training rows. "
                        "Enter at least one completed row for this task before using multitask BO."
                    )

                task_idx = design.task_to_index[active_task]
                fit = fit_multitask_gp(design.X_aug, design.Y[:, 0], task_feature=-1)
                best_f = best_f_for_task(design.Y[:, 0], design.task_indices[:, 0], task_idx=task_idx)

                if q == 1:
                    X_next = suggest_next_mt_ei(
                        fit,
                        bounds=design.bounds,
                        best_f=best_f,
                        task_idx=task_idx,
                        simplex_n=int(design.simplex_n),
                        simplex_bounds=simplex_bounds_full,
                    )
                else:
                    X_next = suggest_q_logei_mt(
                        fit,
                        bounds=design.bounds,
                        best_f=best_f,
                        task_idx=task_idx,
                        q=q,
                        simplex_n=int(design.simplex_n),
                        simplex_bounds=simplex_bounds_full,
                    )

                param_names = design.param_names
                simplex_n = design.simplex_n
                mode = "multitask"

            else:
                # --- single-task fallback ---
                param_names, bounds, simplex_n = infer_bounds_from_last_row(
                    self.df,
                    n_reactants=n_reactants,
                    include_process=True,
                )

                X, Y = build_XY_from_df(self.df, param_names=param_names, y_cols=[objective], dropna=True)
                if X.shape[0] < 3:
                    raise ValueError("Need at least 3 completed rows with objective values to fit a GP.")

                fit = fit_single_task_gp(X, Y[:, 0])
                best_f = float(np.nanmax(Y[:, 0]))

                if q == 1:
                    X_next = suggest_next_ei(fit, bounds=bounds, best_f=best_f, simplex_n=int(simplex_n), simplex_bounds=simplex_bounds_full)
                else:
                    X_next = suggest_q_logei(
                        fit,
                        bounds=bounds,
                        best_f=best_f,
                        q=q,
                        simplex_n=int(simplex_n),
                        simplex_bounds=simplex_bounds_full,
                    )

                mode = "single-task"

            # Normalize to array of shape (q, d_free)
            X_next = np.asarray(X_next, dtype=float)
            if X_next.ndim == 1:
                X_next = X_next[None, :]

            # For simplex variables, the BO engine returns only N-1 free mass fractions.
            # Reconstruct Reactant N MassFrac before writing the proposed row.
            write_param_names = list(param_names)
            X_next_write = X_next
            if int(simplex_n) > 1:
                n_free = int(simplex_n) - 1
                X_next_write = expand_simplex_X(X_next, simplex_n=int(simplex_n))
                write_param_names = (
                    [f"Reactant {i} MassFrac" for i in range(1, int(simplex_n) + 1)]
                    + list(param_names[n_free:])
                )

            added = 0
            for x_next in X_next_write:
                new_row = {c: np.nan for c in self.df.columns}
                for name, val in zip(write_param_names, x_next):
                    if name in self.df.columns:
                        new_row[name] = float(val)

                if "Task" in self.df.columns:
                    task_val = (self.active_task_var.get() or "").strip()
                    if not task_val and len(self.df) > 0:
                        last_task = self.df["Task"].dropna().astype(str)
                        last_task = last_task[last_task.str.strip() != ""]
                        if not last_task.empty:
                            task_val = last_task.iloc[-1].strip()
                    new_row["Task"] = task_val

                if "Reaction Number" in self.df.columns:
                    mx = pd.to_numeric(self.df["Reaction Number"], errors="coerce").max()
                    if np.isfinite(mx):
                        new_row["Reaction Number"] = int(mx) + 1

                # Timestamps (stored but hidden)
                now = self._now_iso()
                if "created_at" in self.df.columns:
                    new_row["created_at"] = now
                if "completed_at" in self.df.columns:
                    new_row["completed_at"] = np.nan

                # Increase iteration number for the proposal (if column exists)
                if "Iteration Number" in self.df.columns:
                    mx_it = pd.to_numeric(self.df["Iteration Number"], errors="coerce").max()
                    if np.isfinite(mx_it):
                        new_row["Iteration Number"] = int(mx_it) + 1

                # Append row
                self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

                # Recompute derived columns and user-facing volume.
                # bo_metrics_v3 is task-aware and uses task-specific MWs when a Task column exists.
                self.df = refresh_table(self.metadata, self.df)
                if "Liquid Additive (uL/mg)" in self.df.columns and "util_mass" in self.df.columns:
                    if "Liquid Additive Volume (uL)" not in self.df.columns:
                        self.df["Liquid Additive Volume (uL)"] = np.nan
                    self.df["Liquid Additive Volume (uL)"] = (
                        pd.to_numeric(self.df["Liquid Additive (uL/mg)"], errors="coerce")
                        * pd.to_numeric(self.df["util_mass"], errors="coerce")
                    )

                # Populate _proposed columns for the newly added row (traceability)
                try:
                    _, _, _, n_reactants2, _ = parse_reactants_from_metadata(self.metadata)
                    n_reactants2 = int(n_reactants2)
                except Exception:
                    n_reactants2 = 0
                new_idx = int(self.df.index.max())
                for i in range(1, n_reactants2 + 1):
                    a = f"Reactant {i} Mass"
                    p = f"Reactant {i} Mass_proposed"
                    if p in self.df.columns and a in self.df.columns:
                        self.df.at[new_idx, p] = self.df.at[new_idx, a]
                for a, p in (("Temperature (°C)", "Temperature (°C)_proposed"), ("Reaction Time (min)", "Reaction Time (min)_proposed")):
                    if a in self.df.columns and p in self.df.columns:
                        self.df.at[new_idx, p] = self.df.at[new_idx, a]
                if "Liquid Additive Volume (uL)" in self.df.columns and "Liquid Additive Volume (uL)_proposed" in self.df.columns:
                    self.df.at[new_idx, "Liquid Additive Volume (uL)_proposed"] = self.df.at[new_idx, "Liquid Additive Volume (uL)"]

                # Fill bounds/range columns for the new row (needed for subsequent BO runs)
                prev_idx = None
                if len(self.df.index) >= 2:
                    prev_idx = int(self.df.index[-2])
                if prev_idx is None:
                    prev_idx = int(self.df.index.min())

                range_like = []
                for c in self.df.columns:
                    cs = str(c)
                    if ("Range Bottom" in cs) or ("Range Top" in cs):
                        range_like.append(c)
                        continue
                    if cs.endswith(" Min (°C)") or cs.endswith(" Max (°C)"):
                        range_like.append(c)
                        continue
                    if cs.endswith(" Min (min)") or cs.endswith(" Max (min)"):
                        range_like.append(c)
                        continue
                    if cs.endswith(" Min (mg/mL)") or cs.endswith(" Max (mg/mL)"):
                        range_like.append(c)
                        continue
                    if cs.endswith(" Min (uL/mg)") or cs.endswith(" Max (uL/mg)"):
                        range_like.append(c)
                        continue

                for c in range_like:
                    cur = self.df.at[new_idx, c]
                    if pd.isna(cur) or cur == "":
                        self.df.at[new_idx, c] = self.df.at[prev_idx, c]

                added += 1

            # Ensure visibility/decimal vars exist for any newly created columns
            for c in map(str, self.df.columns):
                if c not in self.col_visible:
                    self.col_visible[c] = tk.BooleanVar(value=True)
                self._get_decimals_var_for_col(c)

            self._refresh_tree()
            self.refresh_todo_list()
            messagebox.showinfo("Suggestion added", f"Added {added} suggested experiment(s) as new row(s) using {mode} BO.")

        except Exception as e:
            messagebox.showerror("Suggestion failed", str(e))

    def update_nmr_values(self) -> None:
        if self.df is None:
            return
        if self.metadata is None:
            return

        nmr_col = "NMR_conversion" if "NMR_conversion" in self.df.columns else None
        if nmr_col is None:
            messagebox.showerror("Missing column", "No NMR_conversion column found.")
            return

        updated = 0
        for idx, var in self._nmr_entries.items():
            txt = var.get().strip()
            if not txt:
                continue
            try:
                val = float(txt)
            except ValueError:
                messagebox.showerror("Invalid value", f"Row {idx+1}: NMR_conversion must be numeric.")
                return
            self.df.at[idx, nmr_col] = val
            if "completed_at" in self.df.columns:
                self.df.at[idx, "completed_at"] = self._now_iso()
            updated += 1

        if updated:
            self.df = refresh_table(self.metadata, self.df)
            self._refresh_tree()
            self.refresh_todo_list()
            messagebox.showinfo("Updated", f"Updated NMR_conversion for {updated} row(s).")


def main() -> None:
    root = tk.Tk()
    _ = BayesianOptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()