__all__ = [
    'check_floats',
]


def check_floats(
    *floats: float | tuple[str, float],
    min: float | int | None = None,
    max: float | int | None = None,
):
    """Checks whether `floats` is in range of (`min`, `max`).

    Args:
        min (float | int | None, optional): Minimum value. Defaults to None.
        max (float | int | None, optional): Maximum value. Defaults to None.

    Raises:
        ValueError: Raised if a float is not inside the range (min, max).
    """

    for i, value in enumerate(floats):
        if isinstance(value, tuple):
            name, value = value
            name = f'{name} (got: {value})'
        else:
            name = f'float at position {i} (got: {value})'

        if min is not None:
            if min > value:
                raise ValueError(f'{name} must be greater than {min}.')

        if max is not None:
            if max < value:
                raise ValueError(f'{name} must be smaller than {max}.')
