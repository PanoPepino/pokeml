"""Original table helpers, separated from dataset validation."""
import json
import pandas as pd
import numpy as np
from rich.table import Table
from rich import box

# Convert a DataFrame-like into a Rich table


def df_to_table(
    obj,
    title: str = None,
    show_index: bool = False,
    index_name: str | None = None,
    hide_columns: list[str] | None = None,
    parse_json_columns: list[str] | None = None,
    table_box=box.ROUNDED,
    border_style: str = "white",
) -> Table:

    def _looks_like_json_string(x):
        if not isinstance(x, str):
            return False
        s = x.strip()
        return (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}"))

    def _parse_json_columns(df):
        df = df.copy()

        if parse_json_columns is not None:
            cols_to_parse = [col for col in parse_json_columns if col in df.columns]
        else:
            cols_to_parse = [
                col for col in df.columns
                if len(df[col].dropna()) > 0 and df[col].dropna().map(_looks_like_json_string).all()
            ]

        for col in cols_to_parse:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else x
            )

        return df

    def _format_value(v):
        if isinstance(v, dict):
            lines = []
            for k, val in v.items():
                if isinstance(val, (float, np.floating)):
                    lines.append(f"{k}: {float(val):.2f}")
                elif isinstance(val, (int, np.integer)):
                    lines.append(f"{k}: {val}")
                else:
                    lines.append(f"{k}: {val}")
            return "\n".join(lines)

        if isinstance(v, (list, tuple, set, np.ndarray)):
            items = []
            for item in v:
                if isinstance(item, (float, np.floating)) and not pd.isna(item):
                    items.append(f"{float(item):.2f}")
                elif isinstance(item, (int, np.integer)) and not pd.isna(item):
                    items.append(str(item))
                elif pd.isna(item) if not isinstance(item, (list, tuple, set, np.ndarray, dict)) else False:
                    continue
                else:
                    items.append(str(item))
            return "\n".join(items)

        if pd.isna(v) if not isinstance(v, (list, tuple, set, np.ndarray, dict)) else False:
            return ""

        if isinstance(v, (float, np.floating)):
            return f"{float(v):.2f}"
        if isinstance(v, (int, np.integer)):
            return str(v)

        return str(v)

    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name if obj.name is not None else "value")
    else:
        df = obj.copy()

    df = _parse_json_columns(df)

    if hide_columns:
        df = df.drop(columns=hide_columns, errors="ignore")

    t = Table(
        title=title,
        show_header=True,
        header_style="bold red",
        show_lines=True,
        box=table_box,
        border_style=border_style,
    )

    if show_index:
        label = index_name if index_name is not None else (df.index.name or "")
        t.add_column(str(label), overflow="fold")

    for col in df.columns:
        t.add_column(str(col), overflow="fold")

    for idx, row in df.iterrows():
        values = [_format_value(idx)] if show_index else []
        values.extend(_format_value(v) for v in row.values)
        t.add_row(*values)

    return t




