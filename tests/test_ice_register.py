#
# test_ice_register.py
#
# The ice must sit where the atoms are. The simulators translate atoms by
# (offset - origin_x, offset - origin_y, 0) after subtracting the shift, so the
# ice centre must move by the same amounts, and by nothing in z.
#
import numpy as np

from parakeet.simulate.engine import ice_centre


def test_ice_moves_with_the_atoms():
    sample_centre = np.array([1200.0, 1300.0, 250.0])
    offset, origin = 300.0, (100.0, 50.0)
    shift = np.array([5.0, -7.0, 12.0])
    centre = ice_centre(sample_centre, offset, origin, shift)
    atom = (
        sample_centre - shift + np.array([offset - origin[0], offset - origin[1], 0.0])
    )
    assert np.allclose(centre, atom)


def test_ice_has_no_z_offset():
    for offset in (0.0, 150.0, 300.0):
        centre = ice_centre((0.0, 0.0, 500.0), offset, (0.0, 0.0), (0.0, 0.0, 0.0))
        assert centre[2] == 500.0
