"""
Misc utils for generating data
"""
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
import gzip
import tarfile
import logging
import glob

from smart_open import open as smart_open

from gretel_synthetics.batch import DataFrameBatch, MAX_INVALID
# from gretel_synthetics.config import config_from_model_dir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_model_dir_batch_mode(model_dir: str) -> bool:
    path = Path(model_dir)
    if not path.is_dir():
        raise TypeError("model_path must be a directory")

    for sub_path in path.glob("*"):
        if sub_path.name.startswith("batch_"):
            return True

    return False


def archive_model_dir(model_dir: str, dest: str):
    if not dest.endswith(".tar.gz"):
        dest = dest + ".tar.gz"

    with NamedTemporaryFile(suffix=".tar.gz") as fout:
        with tarfile.open(fout.name, mode="w:gz") as tar:
            for fname in glob.glob(model_dir + "/**", recursive=True):
                path = Path(fname)
                if path.suffix in (".csv", ".txt") or path.is_dir():
                    continue
                target = str(Path(path.parent.name) / path.name)
                tar.add(path.resolve(), arcname=target)

        with smart_open(dest, "wb") as sout:
            with open(fout.name, "rb") as sin:
                sout.write(sin.read())


class DataFileGenerator:
    """Utilize existing models to generate new data and save
    to disk.
    """

    raw_model_path: str
    model_path: Path

    def __init__(self, model_path: str):
        self.raw_model_path = model_path
        path = Path(model_path)
        if not path.is_dir() and not path.suffixes == [".tar", ".gz"]:
            raise TypeError("Unrecognized model path, should be a dir or a tar.gz")

        self.model_path = path

    def generate(self, count: int, file_name: str):
        if self.model_path.is_dir():
            return self._generate(self.model_path, count, file_name)

        if self.model_path.suffixes == [".tar", ".gz"]:
            with TemporaryDirectory() as tmpdir:
                with smart_open(str(self.raw_model_path), "rb", ignore_ext=True) as fin:
                    with gzip.open(fin) as gzip_in:
                        with tarfile.open(fileobj=gzip_in, mode="r:gz") as tar_in:
                            logging.info("Extracting archive to temp dir...")
                            tar_in.extractall(tmpdir)
                
                return self._generate(Path(tmpdir), count, file_name)

    def _generate(self, model_dir: Path, count: int, file_name: str) -> str:
        batch_mode = is_model_dir_batch_mode(model_dir)

        if batch_mode:
            out_fname = f"{file_name}.csv"
            batcher = DataFrameBatch(mode="read", checkpoint_dir=str(model_dir))
            batcher.generate_all_batch_lines(num_lines=count, max_invalid=max(count, MAX_INVALID), parallelism=1)
            out_df = batcher.batches_to_df()
            out_df.to_csv(out_fname, index=False)
            return out_fname
