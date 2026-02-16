from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, TextIO

from string import Template

import csv
import json
import math
import os
import re


DELIMITER: str = ","  # matches defs.jl
NEWLINE: str = "\r\n"  # matches defs.jl


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_cell(text: str) -> Any:
    """Parse a single delimited cell.

    The reference reader (readdlm) returns a matrix of ``Any`` and will
    attempt to parse numeric values. We mimic that behavior with a small
    set of conversions:

    * empty -> "" (empty string)
    * true/false -> bool
    * NaN/Inf/-Inf -> floats
    * integers -> int
    * floats -> float
    * otherwise -> original string (stripped)
    """

    s = text.strip()
    if s == "":
        return ""

    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False
    if sl == "nan":
        return math.nan
    if sl in ("inf", "+inf"):
        return math.inf
    if sl == "-inf":
        return -math.inf

    if _INT_RE.match(s):
        try:
            return int(s)
        except ValueError:
            pass
    if _FLOAT_RE.match(s):
        try:
            return float(s)
        except ValueError:
            pass

    return s


@dataclass
class Table:
    """A rectangular table with named columns.

    This mirrors the reference ``Table`` struct in ``file_io.jl``.
    """

    name: str
    header: List[str]
    data: List[List[Any]]

    def __post_init__(self) -> None:
        if len(set(self.header)) != len(self.header):
            raise ValueError(f"Table header contains duplicates: {self.header!r}")
        for row in self.data:
            if len(row) != len(self.header):
                raise ValueError(
                    f"Row length {len(row)} does not match header length {len(self.header)}"
                )

    @property
    def num_rows(self) -> int:
        return len(self.data)

    @property
    def num_cols(self) -> int:
        return len(self.header)

    @property
    def header_dict(self) -> Dict[str, int]:
        return {h: i for i, h in enumerate(self.header)}

    @property
    def columns(self) -> Dict[str, List[Any]]:
        cols: Dict[str, List[Any]] = {h: [] for h in self.header}
        for row in self.data:
            for h, v in zip(self.header, row):
                cols[h].append(v)
        return cols

    def column(self, name: str) -> List[Any]:
        """Return a column by header name."""
        idx = self.header_dict[name]
        return [row[idx] for row in self.data]


def read_tables_from_file(filename: str, *, delim: str = DELIMITER) -> Dict[str, Table]:
    """Read a multi-table delimited file into a dict of :class:`Table` objects."""

    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    with open(filename, "r", newline="") as f:
        rows = list(csv.reader(f, delimiter=delim))
    return read_tables_from_rows(rows)


def read_tables_from_rows(rows: Sequence[Sequence[str]]) -> Dict[str, Table]:
    """Parse tables from already-split CSV rows."""

    tables: Dict[str, Table] = {}
    i = 0
    n = len(rows)

    def row_is_blank(r: Sequence[str]) -> bool:
        return len(r) == 0 or (len(r) >= 1 and (r[0].strip() == ""))

    while i < n:
        # Find start of next table
        while i < n and row_is_blank(rows[i]):
            i += 1
        if i >= n:
            break

        header_row = list(rows[i])
        table_name = header_row[0].strip()
        if table_name == "":
            i += 1
            continue

        num_rows: Optional[int] = None
        num_cols: Optional[int] = None
        if len(header_row) >= 2 and header_row[1].strip() != "":
            num_rows = int(_parse_cell(header_row[1]))
        if len(header_row) >= 3 and header_row[2].strip() != "":
            num_cols = int(_parse_cell(header_row[2]))

        i += 1
        if i >= n:
            raise ValueError(f"Unexpected end of file after table header: {table_name}")

        # Column headers: count until first empty cell
        raw_col_row = list(rows[i])
        cols: List[str] = []
        for cell in raw_col_row:
            if cell.strip() == "":
                break
            cols.append(cell.strip())

        if num_cols is not None and num_cols != len(cols):
            raise ValueError(
                f"Table '{table_name}' column count mismatch: expected {num_cols}, got {len(cols)}"
            )
        if num_cols is None:
            num_cols = len(cols)

        i += 1

        # Data rows until blank separator
        data: List[List[Any]] = []
        while i < n and not row_is_blank(rows[i]):
            raw = list(rows[i])
            # ensure length >= num_cols
            if len(raw) < num_cols:
                raw = raw + [""] * (num_cols - len(raw))
            row_vals = [_parse_cell(raw[j]) for j in range(num_cols)]
            data.append(row_vals)
            i += 1

        if num_rows is not None and num_rows != len(data):
            raise ValueError(
                f"Table '{table_name}' row count mismatch: expected {num_rows}, got {len(data)}"
            )

        table = Table(name=table_name, header=cols, data=data)
        if table.name in tables:
            raise ValueError(f"Duplicate table name: {table.name}")
        tables[table.name] = table

        # Skip separator row (if present)
        while i < n and row_is_blank(rows[i]):
            i += 1

    return tables


def write_tables_to_file(
    filename: str,
    tables: Iterable[Table] | Mapping[str, Table] | Table,
    *,
    delim: str = DELIMITER,
    newline: str = NEWLINE,
    write_num_rows: bool = False,
    write_num_cols: bool = False,
) -> None:
    """Write one or more tables to a file in the reference-compatible format."""

    # Normalize input
    if isinstance(tables, Table):
        table_list: List[Table] = [tables]
    elif isinstance(tables, Mapping):
        table_list = list(tables.values())
    else:
        table_list = list(tables)

    with open(filename, "w", newline="") as f:
        for t in table_list:
            _write_single_table(
                f,
                t,
                delim=delim,
                newline=newline,
                write_num_rows=write_num_rows,
                write_num_cols=write_num_cols,
            )


def _write_single_table(
    f: TextIO,
    table: Table,
    *,
    delim: str,
    newline: str,
    write_num_rows: bool,
    write_num_cols: bool,
) -> None:
    # Table header line with trailing delimiter (matches reference writeDlmLine!)
    parts = [
        table.name,
        str(table.num_rows) if write_num_rows else "",
        str(table.num_cols) if write_num_cols else "",
    ]
    f.write(delim.join(parts) + delim + newline)

    # Column headers with trailing delimiter
    f.write(delim.join(table.header) + delim + newline)

    # Data rows. Use ``csv.writer`` so that values containing the delimiter
    # (notably JSON in the ``attributes`` column) are quoted/escaped.
    writer = csv.writer(
        f,
        delimiter=delim,
        lineterminator=newline,
        quoting=csv.QUOTE_MINIMAL,
    )
    for row in table.data:
        writer.writerow(["" if v is None else v for v in row])

    # Separator blank line (reference writes "" plus delimiter + newline)
    f.write(delim + newline)


def table_rows_field_dicts(table: Table, field_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Return a list of dicts, one per row, selecting the given columns."""

    idxs = [table.header_dict[name] for name in field_names]
    out: List[Dict[str, Any]] = []
    for row in table.data:
        out.append({name: row[j] for name, j in zip(field_names, idxs)})
    return out


def parse_attributes_column(table: Table) -> List[Dict[str, Any]]:
    """Parse a JSON "attributes" column if present.

    The reference implementation treats missing/empty attributes as an empty
    dict.
    """

    cols = table.columns
    if "attributes" not in cols:
        return [{} for _ in range(table.num_rows)]

    attrs: List[Dict[str, Any]] = []
    for cell in cols["attributes"]:
        if cell == "" or cell is None:
            attrs.append({})
        else:
            if not isinstance(cell, str):
                cell = str(cell)
            attrs.append(json.loads(cell))
    return attrs


def join_path_if_not_abs(base_path: str, path: str) -> str:
    """
    If `path` is absolute, return its absolute version.
    Otherwise, join it with `base_path`.
    """
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.join(base_path, path)

def interpolate_string(s: str, context: dict | None = None) -> str:
    """
    Safe string interpolation similar to Julia's string interpolation.
    Supports $VAR syntax.
    """
    if context is None:
        context = os.environ  # or a controlled config dict
    return Template(s).safe_substitute(context)