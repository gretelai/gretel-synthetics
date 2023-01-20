import os
import tarfile
import tempfile
import unittest

from gretel_synthetics.utils.tar_util import (
    _is_within_directory,
    DirectoryCheckException,
    PathTraversalException,
    safe_extractall,
)


class TestTarUtil(unittest.TestCase):
    def create_tar(self, name: str, depth: int = 1) -> str:
        dir_level = "../"
        output = os.path.join(self.temp_dir.name, name)

        payload = os.path.join(self.temp_dir.name, "bash_profile")
        with open(payload, "w", encoding="utf-8") as tmpfile:
            tmpfile.write("hacks")
        path = "/"

        if path and path[-1] != "/":
            path += "/"

        dt_path = f"{dir_level*depth}{path}{os.path.basename(payload)}"

        with tarfile.open(output, "w:gz") as tar:
            tar.add(payload, dt_path)
            tar.close()
        return output

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_happy_path_untar(self):
        good_output = self.create_tar("good.tar.gz", 0)
        with open(good_output, "rb") as gzip_in:
            with tarfile.open(fileobj=gzip_in, mode="r:gz") as tar_in:
                safe_extractall(tar_in, self.temp_dir.name)
                expected_file_path = os.path.join(self.temp_dir.name, "bash_profile")
                self.assertTrue(os.path.exists(expected_file_path))

    def test_malicious_tar(self):
        bad_output = self.create_tar("bad.tar.gz", 1)
        with open(bad_output, "rb") as fin:
            with tarfile.open(fileobj=fin, mode="r:gz") as tar_in:
                with self.assertRaises(PathTraversalException):
                    safe_extractall(tar_in, self.temp_dir.name)

    def test_is_directory_explodes_properly(self):
        with self.assertRaises(DirectoryCheckException):
            _is_within_directory(1, "abc")

    def test_is_directory_is_fine(self):
        self.assertTrue(_is_within_directory("abc", "abc/def"))
        self.assertFalse(_is_within_directory("../123", "abc"))
        self.assertFalse(_is_within_directory("/usr/lib", "/usr/lib64/libfoo.so"))


if __name__ == "__main__":
    unittest.main()
