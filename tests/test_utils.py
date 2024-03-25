import pytest
from etapr.utils import check_floats


@pytest.mark.parametrize(
    'floats,min,max',
    [
        ([-1.0], -2, 2.5),
        ([-1.0, 2.4, -1.99], -2, 2.5),
        ([-1.0, 2.4, 2.5, -2], -2, 2.5),
        ([-1.0, 2.4, -1.99], None, 10),
        ([-1.0, 2.4, -1.99], -10, None),
        ([('a', 0.1)], 0, 1),
        ([('a', 0.1), ('b', 0.9)], 0, 1),
        ([('a', 0.1), ('b', 0.9), ('c', 1.0), ('d', 0)], 0, 1),
        ([('a', -0.1), ('b', -0.9)], None, 1),
        ([('a', 0.1), ('b', 1.1)], 0, None),
    ],
)
def test_check_floats(
    floats: list[float | tuple[str, float]],
    min: float | None,
    max: float | None,
):
    check_floats(*floats, min=min, max=max)


@pytest.mark.parametrize(
    'floats,min,max,msg',
    [
        (
            [-1.0, 2.4],
            0,
            2.5,
            'float at position 0 (got: -1.0) must be greater than 0.',
        ),
        (
            [-1.0, 2.4],
            -2,
            1,
            'float at position 1 (got: 2.4) must be smaller than 1.',
        ),
        (
            [('a', -1.1), ('b', 1.1)],
            0,
            1,
            'a (got: -1.1) must be greater than 0.',
        ),
        (
            [('a', -15.2), ('b', 10.5)],
            None,
            1,
            'b (got: 10.5) must be smaller than 1.',
        ),
    ],
)
def test_check_floats_fails(
    floats: list[float | tuple[str, float]],
    min: float | None,
    max: float | None,
    msg: str,
):

    with pytest.raises(ValueError) as exc_info:
        check_floats(*floats, min=min, max=max)

    assert exc_info.value.args[0] == msg
