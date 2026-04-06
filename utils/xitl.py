"""
XiTL (Expert-in-the-Loop) editor for timeseries datasets.

Schema: time_column (datetime), identifier_columns (list[str]), value_column (float).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class XiTLValueError(ValueError):
    """Raised for invalid identifier, time_range, or no matching rows."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChangelogEntry:
    """A single mutation recorded in the changelog."""

    identifier_filter: dict
    time_range: Optional[tuple]
    mutation_type: Literal["set", "add", "subtract", "add_pct", "subtract_pct"]
    value: float


@dataclass
class XitlResult:
    """Result of commit(): final dataframe and changelog."""

    final_df: pd.DataFrame
    changelog: list[ChangelogEntry]
    summary: Optional[dict] = None


# ---------------------------------------------------------------------------
# XiTLSlice
# ---------------------------------------------------------------------------


class XiTLSlice:
    """View of a slice of XiTL data. Mutations apply to the parent XiTL."""

    def __init__(
        self,
        parent: "XiTL",
        identifier_filter: dict,
        time_range: Optional[tuple],
    ):
        self._parent = parent
        self._identifier_filter = identifier_filter
        self._time_range = time_range
        self._mask: Optional[pd.Series] = None

    def _get_mask(self) -> pd.Series:
        """Compute and cache boolean mask for rows matching this slice."""
        if self._mask is not None:
            return self._mask
        df = self._parent._working_df
        mask = pd.Series(True, index=df.index)
        for col, val in self._identifier_filter.items():
            mask &= df[col] == val
        if self._time_range is not None:
            start, end = self._time_range
            tc = self._parent._time_column
            times = pd.to_datetime(df[tc])
            mask &= (times >= start) & (times < end)
        self._mask = mask
        return mask

    def mutate(
        self,
        value: Optional[float] = None,
        delta: Optional[float] = None,
        pct: Optional[float] = None,
        mode: Optional[Literal["set", "add", "subtract", "add_pct", "subtract_pct"]] = None,
    ) -> "XiTLSlice":
        """
        Apply a mutation to this slice and auto-stage to changelog.

        Modes:
        - Set: mutate(value=100) or mutate(mode="set", value=100)
        - Add/Subtract fixed: mutate(delta=10) or mutate(delta=-5)
        - Add/Subtract pct: mutate(pct=10) or mutate(pct=-5)

        Returns self for chaining.
        """
        mask = self._get_mask()
        vc = self._parent._value_column
        df = self._parent._working_df

        # Determine mutation_type and val
        if mode is not None:
            mutation_type = mode
            if mutation_type == "set":
                if value is None:
                    raise XiTLValueError("mode='set' requires value=")
                val = float(value)
            elif mutation_type in ("add", "subtract"):
                if delta is None:
                    raise XiTLValueError("mode='add'/'subtract' requires delta=")
                val = abs(float(delta))
            elif mutation_type in ("add_pct", "subtract_pct"):
                if pct is None:
                    raise XiTLValueError(
                        "mode='add_pct'/'subtract_pct' requires pct="
                    )
                val = abs(float(pct))
            else:
                raise XiTLValueError(f"Unknown mode: {mode}")
        else:
            if value is not None:
                mutation_type = "set"
                val = float(value)
            elif delta is not None:
                mutation_type = "add" if delta >= 0 else "subtract"
                val = abs(float(delta))
            elif pct is not None:
                mutation_type = "add_pct" if pct >= 0 else "subtract_pct"
                val = abs(float(pct))
            else:
                raise XiTLValueError(
                    "mutate() requires one of: value=, delta=, or pct="
                )

        # Apply transformation
        if mutation_type == "set":
            df.loc[mask, vc] = val
        elif mutation_type == "add":
            df.loc[mask, vc] = df.loc[mask, vc] + val
        elif mutation_type == "subtract":
            df.loc[mask, vc] = df.loc[mask, vc] - val
        elif mutation_type == "add_pct":
            df.loc[mask, vc] = df.loc[mask, vc] * (1 + val / 100)
        elif mutation_type == "subtract_pct":
            df.loc[mask, vc] = df.loc[mask, vc] * (1 - val / 100)
        else:
            raise XiTLValueError(f"Unknown mutation_type: {mutation_type}")

        # Invalidate mask cache (values changed, mask indices still valid)
        self._mask = None

        # Auto-stage
        self._parent._changelog.append(
            ChangelogEntry(
                identifier_filter=dict(self._identifier_filter),
                time_range=self._time_range,
                mutation_type=mutation_type,
                value=val,
            )
        )
        return self

    def preview(
        self,
        identifier: Optional[Union[tuple, dict]] = None,
        time_range: Optional[Union[tuple, slice, None]] = None,
        max_series: int = 4,
    ) -> plt.Figure:
        """
        Preview original vs mutated timeseries in a two-panel matplotlib figure.

        Default: use current slice's identifier and time_range.
        Override: pass identifier and/or time_range.
        """
        parent = self._parent
        id_cols = parent._identifier_columns
        tc = parent._time_column
        vc = parent._value_column

        # Resolve identifier and time_range for preview
        if identifier is not None or time_range is not None:
            id_filter = parent._resolve_identifier(
                identifier if identifier is not None else self._identifier_filter
            )
            tr = parent._resolve_time_range(
                time_range if time_range is not None else self._time_range
            )
        else:
            id_filter = self._identifier_filter
            tr = self._time_range

        # Get series to plot (may be partial dict -> multiple series)
        df_orig = parent._original_df
        df_mutated = parent._working_df

        # Filter by identifier
        mask_id = pd.Series(True, index=df_orig.index)
        for col, val in id_filter.items():
            mask_id &= df_orig[col] == val
        filtered_orig = df_orig[mask_id].copy()
        filtered_mut = df_mutated[mask_id].copy()

        if tr is not None:
            start, end = tr
            times_orig = pd.to_datetime(filtered_orig[tc])
            times_mut = pd.to_datetime(filtered_mut[tc])
            filtered_orig = filtered_orig[(times_orig >= start) & (times_orig < end)]
            filtered_mut = filtered_mut[(times_mut >= start) & (times_mut < end)]

        # Get distinct series (full identifier combos in filtered data)
        cols_present = [c for c in id_cols if c in filtered_orig.columns]
        if not cols_present:
            cols_present = id_cols
        series_list = (
            filtered_mut.drop_duplicates(subset=cols_present)[cols_present]
            .head(max_series)
            .to_dict("records")
        )

        if not series_list:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].text(0.5, 0.5, "No data to preview", ha="center", va="center")
            ax[1].text(0.5, 0.5, "No data to preview", ha="center", va="center")
            return fig

        n = len(series_list)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)

        for i, sf in enumerate(series_list):
            # Build full filter (partial id_filter + series values for remaining cols)
            full_filter = {**id_filter}
            for c in cols_present:
                if c not in full_filter:
                    full_filter[c] = sf[c]

            mask_orig = pd.Series(True, index=filtered_orig.index)
            mask_mut = pd.Series(True, index=filtered_mut.index)
            for col, val in full_filter.items():
                if col in filtered_orig.columns:
                    mask_orig &= filtered_orig[col] == val
                if col in filtered_mut.columns:
                    mask_mut &= filtered_mut[col] == val

            before = filtered_orig[mask_orig].sort_values(tc)
            after = filtered_mut[mask_mut].sort_values(tc)

            label = " | ".join(str(v) for v in full_filter.values())

            ax_left = axes[i, 0]
            ax_right = axes[i, 1]

            if not before.empty:
                ax_left.plot(
                    pd.to_datetime(before[tc]),
                    before[vc],
                    color="tab:orange",
                    linewidth=1.2,
                    linestyle="--",
                    label="Original",
                )
            if not after.empty:
                ax_right.plot(
                    pd.to_datetime(after[tc]),
                    after[vc],
                    color="tab:blue",
                    linewidth=1,
                    label="Mutated",
                )

            ax_left.set_title(f"Original — {label}", fontsize=11)
            ax_right.set_title(f"Mutated — {label}", fontsize=11)
            ax_left.set_ylabel(vc)
            ax_right.set_ylabel(vc)
            ax_left.legend(fontsize=8)
            ax_right.legend(fontsize=8)
            ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax_right.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        axes[-1, 0].set_xlabel(tc)
        axes[-1, 1].set_xlabel(tc)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# XiTL
# ---------------------------------------------------------------------------


class XiTL:
    """Expert-in-the-loop editor for timeseries datasets."""

    def __init__(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_column: str,
        identifier_columns: list[str],
    ):
        """
        Initialize XiTL with a timeseries dataframe.

        Args:
            df: Timeseries data.
            time_column: Name of datetime column.
            value_column: Name of numeric value column.
            identifier_columns: Names of columns that identify a series.
        """
        df = df.copy()
        self._time_column = time_column
        self._value_column = value_column
        self._identifier_columns = list(identifier_columns)

        # Validate schema
        required = [time_column, value_column] + self._identifier_columns
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise XiTLValueError(f"Missing columns: {missing}")

        try:
            pd.to_datetime(df[time_column])
        except Exception as e:
            raise XiTLValueError(
                f"time_column '{time_column}' must be datetime-like: {e}"
            )

        try:
            is_numeric = np.issubdtype(df[value_column].dtype, np.number)
        except (TypeError, AttributeError):
            is_numeric = False
        if not is_numeric:
            raise XiTLValueError(
                f"value_column '{value_column}' must be numeric"
            )

        self._original_df = df.copy()
        self._working_df = df.copy()
        self._changelog: list[ChangelogEntry] = []

    def _resolve_identifier(self, identifier: Union[tuple, dict]) -> dict:
        """Resolve identifier (tuple or partial dict) to a filter dict."""
        id_cols = self._identifier_columns
        if isinstance(identifier, (list, tuple)):
            if len(identifier) != len(id_cols):
                raise XiTLValueError(
                    f"Identifier tuple length {len(identifier)} must match "
                    f"identifier_columns length {len(id_cols)}"
                )
            return dict(zip(id_cols, identifier))
        if isinstance(identifier, dict):
            bad = [k for k in identifier if k not in id_cols]
            if bad:
                raise XiTLValueError(
                    f"Identifier keys {bad} not in identifier_columns {id_cols}"
                )
            return dict(identifier)
        raise XiTLValueError(
            f"identifier must be tuple or dict, got {type(identifier)}"
        )

    def _resolve_time_range(
        self,
        time_range: Optional[Union[tuple, slice]],
    ) -> Optional[tuple]:
        """Resolve time_range to (start, end) or None."""
        if time_range is None:
            return None
        if isinstance(time_range, slice):
            return (time_range.start, time_range.stop)
        if isinstance(time_range, (tuple, list)) and len(time_range) >= 2:
            return (time_range[0], time_range[1])
        raise XiTLValueError(
            f"time_range must be (start, end), slice(start, end), or None, "
            f"got {type(time_range)}"
        )

    def _build_mask(
        self,
        identifier_filter: dict,
        time_range: Optional[tuple],
    ) -> pd.Series:
        """Build boolean mask for identifier_filter and time_range."""
        df = self._working_df
        mask = pd.Series(True, index=df.index)
        for col, val in identifier_filter.items():
            mask &= df[col] == val
        if time_range is not None:
            start, end = time_range
            times = pd.to_datetime(df[self._time_column])
            mask &= (times >= start) & (times < end)
        return mask

    def query(
        self,
        identifier: Union[tuple, dict],
        time_range: Optional[Union[tuple, slice]] = None,
    ) -> XiTLSlice:
        """
        Query a slice of the dataframe by identifier and time_range.

        identifier: Tuple in column order or partial dict.
        time_range: (start, end) [start incl, end excl], slice(start, end), or None.

        Returns XiTLSlice. Raises XiTLValueError if invalid or no match.
        """
        id_filter = self._resolve_identifier(identifier)
        tr = self._resolve_time_range(time_range)

        # Validate: at least one row matches
        mask = self._build_mask(id_filter, tr)
        if not mask.any():
            raise XiTLValueError(
                f"No rows match identifier {id_filter}"
                + (f" and time_range {tr}" if tr else "")
            )

        return XiTLSlice(parent=self, identifier_filter=id_filter, time_range=tr)

    def commit(self) -> XitlResult:
        """
        Return the final dataframe and changelog.

        _working_df already has all mutations applied.
        """
        n_entries = len(self._changelog)
        summary = {"changelog_entries": n_entries}
        return XitlResult(
            final_df=self._working_df.copy(),
            changelog=list(self._changelog),
            summary=summary,
        )
