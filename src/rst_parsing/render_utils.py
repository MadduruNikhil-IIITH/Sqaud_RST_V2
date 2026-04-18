from __future__ import annotations

import contextlib
import io
from pathlib import Path


def render_quiet(rs3_path: Path, colab: bool = False) -> str:
    import isanlp_rst

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        isanlp_rst.render(str(rs3_path), colab=colab)
    return buf.getvalue()
