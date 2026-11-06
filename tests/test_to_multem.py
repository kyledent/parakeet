#
# test_to_multem.py
#
# AtomData.to_multem builds the engine's atom list from columns. It must give
# exactly the atoms the per-atom tuple construction gave.
#
import numpy as np
import pytest

multem = pytest.importorskip("multem")
if not hasattr(multem.AtomList, "from_arrays"):
    pytest.skip(
        "this python-multem has no AtomList.from_arrays", allow_module_level=True
    )

from parakeet.sample import AtomData  # noqa: E402


def atoms(n, seed=0):
    rng = np.random.default_rng(seed)
    return AtomData(
        atomic_number=rng.integers(1, 80, n),
        x=rng.uniform(0, 1000, n),
        y=rng.uniform(0, 1000, n),
        z=rng.uniform(0, 500, n),
        sigma=rng.uniform(0, 0.2, n),
        occupancy=rng.uniform(0.5, 1, n),
        charge=rng.integers(-2, 3, n),
        group=rng.integers(0, 5, n),
    )


def by_tuples(a):
    d = a.data
    return multem.AtomList(
        zip(
            d["atomic_number"].astype("uint8"),
            d["x"].astype("float32"),
            d["y"].astype("float32"),
            d["z"].astype("float32"),
            d["sigma"].astype("float32"),
            d["occupancy"].astype("float32"),
            [int(0) for i in range(d.shape[0])],
            d["charge"].astype("int8"),
        )
    )


def test_same_atoms_as_the_tuple_construction():
    a = atoms(2000)
    got = a.to_multem()
    ref = by_tuples(a)
    assert len(got) == len(ref) == 2000
    assert all(x == y for x, y in zip(ref, got))


def test_negative_charges_survive():
    a = atoms(200, seed=3)
    got = a.to_multem()
    assert sorted({at[7] for at in got}) == sorted(set(a.data["charge"].astype(int)))


def test_empty():
    assert len(atoms(0).to_multem()) == 0
