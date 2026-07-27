"""Guard the promise that ``import vizlib`` stays fast.

matplotlib and seaborn are installed by default (they are regular runtime
dependencies), but they are a slow import. Because the plotting layer lives
in ``vizlib.plots`` and is not re-exported from ``__init__.py``, a bare
``import vizlib`` must not pull them in — they load only when the user
explicitly does ``from vizlib import plots``. This test locks that in.
"""

import subprocess
import sys


def test_bare_import_does_not_load_plotting_backends():
    """A fresh interpreter importing vizlib must not load matplotlib/seaborn.

    Run in a subprocess so the check is unaffected by other tests in this
    session that legitimately import the plotting backends.
    """
    code = (
        "import sys, vizlib;"
        "assert 'matplotlib' not in sys.modules, 'matplotlib leaked into import vizlib';"
        "assert 'seaborn' not in sys.modules, 'seaborn leaked into import vizlib';"
        "assert vizlib.__version__ == '0.6.2'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
