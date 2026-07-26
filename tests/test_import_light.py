"""Guard the promise that ``import vizlib`` stays pandas-only and fast."""

import subprocess
import sys


def test_bare_import_pulls_no_plotting_libs():
    """A fresh interpreter importing vizlib must not load matplotlib/seaborn.

    Run in a subprocess so the check is unaffected by other tests in this
    session that legitimately import the plotting backends.
    """
    code = (
        "import sys, vizlib;"
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked into import vizlib';"
        "assert 'seaborn' not in sys.modules, 'seaborn leaked into import vizlib';"
        "assert vizlib.__version__ == '0.2.0'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
