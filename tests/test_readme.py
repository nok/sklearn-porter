import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts' / 'update_readme.py'


def test_support_table_is_up_to_date():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), '--check'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
