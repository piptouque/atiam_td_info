
from typing import Any


def log_error(expected: Any, answer: Any, fail: bool = True) -> None:
    print("-- ERROR --")
    print(f"expected:  {expected}")
    print(f"got:       {answer}")
    if fail:
        raise ArgumentError("Failed")
