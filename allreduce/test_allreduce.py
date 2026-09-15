import numpy as np
import pytest
from allreduce.ring_allreduce import ring_allreduce


@pytest.mark.parametrize ( "N, D", [ ( 2, 4 ), ( 3, 12 ), ( 4, 12 ), ( 5, 13 ), ( 8, 100 ) ] )
def test_ring_allreduce_matches_sum ( N, D ) :
    rng = np.random.default_rng ( 0 )
    nodes = [ rng.standard_normal ( D ) for _ in range ( N ) ]
    expected = sum ( nodes )
    out = ring_allreduce ( [ x.copy() for x in nodes ] )
    assert len ( out ) == N
    for y in out :
        np.testing.assert_allclose ( y, expected, atol = 1e-12 )


def test_inputs_not_required_to_survive () :
    """The function may work in place; the caller passes copies if it cares."""
    nodes = [ np.ones ( 6 ) * i for i in range ( 3 ) ]
    out = ring_allreduce ( nodes )
    np.testing.assert_allclose ( out[0], np.ones ( 6 ) * 3 )
