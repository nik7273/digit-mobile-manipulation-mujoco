"""Keep the documented script commands usable from outside the checkout."""

import subprocess
import sys

import pytest

from digit_mujoco.paths import ROOT


@pytest.mark.parametrize("script", ["demo", "standing_hold", "pickup", "carry", "transfer"])
def test_script_help_outside_repository(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / f"{script}.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--headless" in result.stdout
