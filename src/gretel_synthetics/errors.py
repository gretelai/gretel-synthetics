"""
Custom error classes
"""


class GenerationError(Exception):
    pass


class TooManyInvalidError(RuntimeError):
    pass
