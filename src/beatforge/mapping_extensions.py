"""Encoding helpers for the Mapping Extensions integer map format."""
from __future__ import annotations

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def is_int32(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and _INT32_MIN <= value <= _INT32_MAX)


def is_coordinate(value: object, *, mapping_extensions: bool, axis: str = "x",
                   wall: bool = False) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if mapping_extensions:
        return is_int32(value) and not (wall and axis == "y" and value < 0)
    limit = 4 if axis == "x" else 3
    return value in range(limit)


def decode_position(value: int) -> float:
    """Decode an ME lane/layer value to its logical grid coordinate."""
    if not isinstance(value, int) or isinstance(value, bool):
        return float("nan")
    if value >= 1000:
        return (value - 1000) / 1000.0
    if value <= -1000:
        return (value + 1000) / 1000.0
    return float(value)


def direction_bucket(value: int, *, mapping_extensions: bool) -> int | None:
    """Map V3 and Mapping Extensions directions into eight diagnostic sectors."""
    if 0 <= value < 8:
        return value
    if value == 8:
        return None
    if mapping_extensions and 1000 <= value <= 1360:
        degrees = value - 1000
    elif mapping_extensions and 2000 <= value <= 2360:
        degrees = value - 2000
    else:
        return None
    return int((degrees + 22.5) // 45) % 8


def valid_direction(value: object, *, mapping_extensions: bool) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return (0 <= value <= 8 or mapping_extensions and
            (1000 <= value <= 1360 or 2000 <= value <= 2360))


def decode_wall_width(value: int) -> float:
    if value <= -1000:
        return (value + 1000) / 1000.0
    if value >= 1000:
        return (value - 1000) / 1000.0
    return float(value)


def decode_wall_height(value: int) -> float:
    if value <= -1000:
        return (value + 2000) / 1000.0
    if value >= 1000:
        return (value - 1000) / 1000.0
    return float(value)


def valid_wall_dimension(value: object, *, mapping_extensions: bool,
                         height: bool = False) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    if not mapping_extensions:
        return value >= 1
    if not is_int32(value):
        return False
    logical = decode_wall_height(value) if height else decode_wall_width(value)
    return logical != 0 and (not height or logical > 0)
