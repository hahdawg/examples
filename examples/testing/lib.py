"""
This module represents pure library code.
"""
from collections import namedtuple
from typing import Tuple

DbOutput = namedtuple("DbOutput", ["colname", "colvals"])


def process_data(data: DbOutput) -> Tuple[str, str]:
    return (data.colname, "".join(data.colvals))
