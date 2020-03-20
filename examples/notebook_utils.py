import logging
from pathlib import Path
import os


def set_working_dir() -> str:
    # Change to project src directory
    working_dir = Path.cwd()
    if working_dir.parts[-1] == 'examples':
        working_dir = Path(Path.cwd()).parents[0] / 'src'
        os.chdir(working_dir)

    print(working_dir.resolve())
    return working_dir.resolve()
