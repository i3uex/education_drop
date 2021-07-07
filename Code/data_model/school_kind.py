from collections import namedtuple
from enum import Enum

SchoolKindItem = namedtuple(
    "SchoolKindItem",
    ["value", "prefix"])


class SchoolKind(SchoolKindItem, Enum):
    """
    School types available.
    """

    Polytechnic = SchoolKindItem(
        value="Polytechnic",
        prefix="polytechnic"
    )

    Teaching = SchoolKindItem(
        value="Teaching",
        prefix="teaching"
    )

    def __str__(self):
        return self.value

    @staticmethod
    def argparse(value: str):
        try:
            return SchoolKind[value]
        except KeyError:
            return value
