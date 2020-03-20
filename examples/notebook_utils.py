import logging
from pathlib import Path
import os


def set_working_dir() -> str:
    # Change to project src directory
    current_path = Path.cwd()
    if current_path.parts[-1] == 'examples':
        working_dir = Path(Path.cwd()).parents[0]
        src_dir = working_dir / 'src'
        os.chdir(src_dir)
        print(working_dir.resolve())
    return working_dir.resolve()
