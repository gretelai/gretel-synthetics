import os
import tarfile


class PathTraversalException(Exception):
    pass


class DirectoryCheckException(Exception):
    pass


def _is_within_directory(directory: str, target: str) -> bool:
    try:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)

        prefix = os.path.commonpath([abs_directory, abs_target])

        return prefix == abs_directory
    except Exception as ex:
        raise DirectoryCheckException("Unable to safely check tarball") from ex


def safe_extractall(
    tar: tarfile.TarFile,
    path: str = ".",
    *,
    numeric_owner: bool = False,
) -> None:

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not _is_within_directory(path, member_path):
            raise PathTraversalException("Path Traversal Exploit is not allowed")

    tar.extractall(path, numeric_owner=numeric_owner)
