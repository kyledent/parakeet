#
# test_regions_render_gpu.py
#
# Regions of material rendered through the multislice engine: each point gets
# the material of the one region that owns it, a material's mean phase scales
# with its density, a nested region displaces its parent rather than adding to
# it, and a cylinder renders along any axis. Needs a CUDA GPU and an engine
# with a multi-region masker; skipped otherwise (as in CI).
#
# Phases are measured against stated predictions, with a tolerance of five
# standard errors estimated from 8 x 8 pixel blocks of the same image.
#
import numpy as np
import pytest
import yaml

import parakeet.io
import parakeet.sample
import parakeet.simulate
import parakeet.simulate.engine
from parakeet.engine_check import EngineFeatureError, engine_features, import_multem

multem = import_multem()
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        multem is None
        or not getattr(multem, "is_gpu_available", lambda: False)()
        or not engine_features(multem)["multi_region_masker"]
        or not engine_features(multem)["grf_seed"],
        reason="needs a CUDA GPU and an engine with a multi-region masker",
    ),
]

NX = 256
ANGPIX = 1.5
FIELD = NX * ANGPIX  # 384 A
THICKNESS = 120.0
MARGIN = 20  # pixels; the exit wave keeps them
FAR = 2000.0  # A: regions reach past the simulation grid so their edges do not
#               fringe into the field
WATER_DENSITY = 0.94
DENSE = {"density": 2 * WATER_DENSITY}  # twice the molecules, twice the mean


def doc(*regions, medium="vacuum"):
    return {"schema": "parakeet-regions/1", "medium": medium, "regions": list(regions)}


def box(
    id, x0, x1, material="water", y0=-FAR, y1=FIELD + FAR, z0=0.0, z1=THICKNESS, **kw
):
    return dict(
        id=id,
        kind="box",
        material=material,
        centre_A=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
        half_lengths_A=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2],
        **kw,
    )


def render_with_ice(directory, ice_parameters, seed=1234):
    """render() of plain ice with extra ice_parameters (no regions)"""
    return render(directory, None, seed=seed, ice_parameters=ice_parameters)


def render(
    directory, regions, materials=None, seed=1234, ice_parameters=None, origin=(0, 0)
):
    directory.mkdir()
    # MULTEM slices only where there are atoms: two hydrogen atoms near a
    # corner of the field, one just inside each face of the slab
    anchors = directory / "anchors.pdb"
    anchors.write_text(
        "".join(
            "HETATM%5d  H   HOH A%4d    %8.3f%8.3f%8.3f  1.00  0.00           H\n"
            % (i, i, FIELD - 6, FIELD - 6, z)
            for i, z in enumerate((1.0, THICKNESS - 1.0), start=1)
        )
        + "END\n"
    )
    sample = {
        "coords": {"filename": str(anchors), "recentre": False},
        "box": [FIELD, FIELD, THICKNESS],
        "centre": [FIELD / 2, FIELD / 2, THICKNESS / 2],
        "shape": {
            "type": "cuboid",
            "cuboid": {"length_x": FIELD, "length_y": FIELD, "length_z": THICKNESS},
        },
    }
    if regions is not None:
        sample["regions"] = {"document": regions}
    config = {
        "microscope": {
            "beam": {"energy": 300, "electrons_per_angstrom": 30.0},
            "detector": {
                "nx": NX,
                "ny": NX,
                "pixel_size": ANGPIX,
                "origin": list(origin),
            },
        },
        "sample": sample,
        "scan": {"mode": "still", "num_images": 1},
        "simulation": {
            "ice": True,
            "seed": seed,
            "margin": MARGIN,
            "padding": 20,
            "materials": materials,
            **({"ice_parameters": ice_parameters} if ice_parameters else {}),
        },
        "multiprocessing": {"device": "gpu", "nproc": 1, "gpu_id": [0]},
    }
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    parakeet.sample.new(str(path), str(directory / "sample.h5"))
    parakeet.simulate.exit_wave(
        str(path), str(directory / "sample.h5"), str(directory / "exit_wave.h5")
    )
    reader = parakeet.io.open(str(directory / "exit_wave.h5"))
    return np.asarray(reader.data[0], dtype=np.complex128), path


def px(x):
    """The exit-wave pixel of a sample coordinate (A); the margin is kept"""
    return int(round(x / ANGPIX)) + MARGIN


def window(x0, x1, y0, y1):
    """Rows (y) and columns (x) of a window given in sample coordinates (A)"""
    return slice(px(y0), px(y1)), slice(px(x0), px(x1))


def phase_against(psi, reference):
    """The phase relative to the mean wave over a reference window"""
    ref = psi[reference].mean()
    return np.angle(psi * np.conj(ref / abs(ref)))


def mean_and_se(values, b=8):
    """The mean of a window, and its standard error from b x b block means"""
    ny, nx = (values.shape[0] // b) * b, (values.shape[1] // b) * b
    blocks = values[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
    return float(values.mean()), float(blocks.std(ddof=1) / np.sqrt(blocks.size))


def masked_mean_and_se(values, mask, b=8):
    """As mean_and_se, over the pixels of a mask, with blocks that are inside it"""
    ny, nx = (values.shape[0] // b) * b, (values.shape[1] // b) * b
    v = values[:ny, :nx].reshape(ny // b, b, nx // b, b)
    m = mask[:ny, :nx].reshape(ny // b, b, nx // b, b)
    full = m.all(axis=(1, 3))
    blocks = v.mean(axis=(1, 3))[full]
    return float(values[mask].mean()), float(blocks.std(ddof=1) / np.sqrt(blocks.size))


def test_the_ice_path_and_a_water_medium_with_no_regions_agree(tmp_path):
    old, _ = render(tmp_path / "ice", None)
    new, _ = render(tmp_path / "medium", doc(medium="water"))
    whole = (slice(MARGIN, MARGIN + NX), slice(MARGIN, MARGIN + NX))
    a = np.angle(old[whole] / old[whole].mean())
    b = np.angle(new[whole] / new[whole].mean())
    print(
        "ice path vs water medium: phase s.d. %.5f vs %.5f; field difference %.3g relative"
        % (a.std(), b.std(), np.abs(old - new).max() / np.abs(old).max())
    )
    assert b.std() == pytest.approx(a.std(), rel=0.05)
    assert abs(new[whole].mean()) == pytest.approx(abs(old[whole].mean()), rel=0.01)
    assert np.angle(new[whole].mean() / old[whole].mean()) == pytest.approx(0, abs=0.02)


def test_three_materials_side_by_side(tmp_path):
    # Water for x < 128 A, a twice-as-dense material for 128 <= x < 256, and
    # vacuum (the medium) beyond
    psi, _ = render(
        tmp_path / "three",
        doc(box("water", -FAR, 128.0), box("dense", 128.0, 256.0, material="dense")),
        materials={"dense": DENSE},
    )
    vacuum = window(280, 370, 0, FIELD)
    phase = phase_against(psi, vacuum)
    water, se_water = mean_and_se(phase[window(10, 110, 0, FIELD)])
    dense, se_dense = mean_and_se(phase[window(146, 238, 0, FIELD)])
    empty = phase[vacuum]
    ratio = dense / water
    se_ratio = ratio * np.hypot(se_water / water, se_dense / dense)
    print(
        "water %.4f +- %.4f rad, dense %.4f +- %.4f rad, ratio %.4f +- %.4f; "
        "vacuum s.d. %.2g rad"
        % (water, se_water, dense, se_dense, ratio, se_ratio, empty.std())
    )
    assert water > 10 * se_water
    assert ratio == pytest.approx(2.0, abs=max(5 * se_ratio, 0.02))
    assert empty.std() < 0.05 * phase[window(10, 110, 0, FIELD)].std()


def test_a_nested_region_displaces_its_parent(tmp_path):
    # A dense sphere inside a water slab: through the sphere the path is part
    # water and part dense, never both
    radius, centre = 50.0, (FIELD / 2, FIELD / 2, THICKNESS / 2)
    psi, _ = render(
        tmp_path / "nested",
        doc(
            box("slab", -FAR, 300.0),
            dict(
                id="ball",
                kind="sphere",
                material="dense",
                parent="slab",
                centre_A=list(centre),
                radius_A=radius,
            ),
        ),
        materials={"dense": DENSE},
    )
    phase = phase_against(psi, window(320, 380, 0, FIELD))
    slab, se_slab = mean_and_se(phase[window(10, 100, 0, FIELD)])

    # The disk within 30 A of the sphere's axis, and the chord through it
    rows, cols = np.mgrid[0 : psi.shape[0], 0 : psi.shape[1]]
    x = (cols - MARGIN) * ANGPIX
    y = (rows - MARGIN) * ANGPIX
    rho2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
    disk = rho2 < 30.0**2
    chord = 2 * np.sqrt(np.clip(radius**2 - rho2[disk], 0, None)).mean()
    ball, se_ball = masked_mean_and_se(phase, disk)

    # Phase per A of water is slab / THICKNESS; the sphere's material has twice
    # that, and replaces the water along the chord
    displaced = 1 + chord / THICKNESS
    added = 1 + 2 * chord / THICKNESS
    measured = ball / slab
    se = measured * np.hypot(se_ball / ball, se_slab / slab)
    print(
        "slab %.4f rad, ball %.4f rad (chord %.1f A): ratio %.4f +- %.4f; "
        "displaced predicts %.4f, added %.4f"
        % (slab, ball, chord, measured, se, displaced, added)
    )
    assert measured == pytest.approx(displaced, abs=max(5 * se, 0.02))
    assert abs(measured - added) > 10 * se


def test_cylinders_render_along_any_axis(tmp_path):
    # A water box for reference; a cylinder along the beam (z); one across it (y)
    zr, zc = 50.0, (200.0, 100.0)
    yr, yx = 40.0, 300.0
    psi, _ = render(
        tmp_path / "cylinders",
        doc(
            box("ref", -FAR, 100.0),
            dict(
                id="along_beam",
                kind="cylinder",
                material="water",
                centre_A=[zc[0], zc[1], THICKNESS / 2],
                axis=[0, 0, 1],
                radius_A=zr,
                length_A=THICKNESS,
            ),
            dict(
                id="across_beam",
                kind="cylinder",
                material="water",
                centre_A=[yx, FIELD / 2, THICKNESS / 2],
                axis=[0, 1, 0],
                radius_A=yr,
                length_A=FIELD + 2 * FAR,
            ),
        ),
    )
    phase = phase_against(psi, window(110, 145, 200, 380))
    ref, se_ref = mean_and_se(phase[window(10, 90, 0, FIELD)])

    rows, cols = np.mgrid[0 : psi.shape[0], 0 : psi.shape[1]]
    x = (cols - MARGIN) * ANGPIX
    y = (rows - MARGIN) * ANGPIX
    disk = (x - zc[0]) ** 2 + (y - zc[1]) ** 2 < 30.0**2
    along, se_along = masked_mean_and_se(phase, disk)

    strip = (np.abs(x - yx) < 10.0) & (y > 20) & (y < FIELD - 20)
    path = (2 * np.sqrt(yr**2 - (x[strip] - yx) ** 2)).mean()
    across, se_across = masked_mean_and_se(phase, strip, b=4)

    print(
        "reference %.4f rad; along the beam %.4f (expects %.4f); across %.4f "
        "(expects %.4f, path %.1f A)"
        % (ref, along, ref, across, ref * path / THICKNESS, path)
    )
    assert along / ref == pytest.approx(1.0, abs=max(5 * se_along / ref, 0.03))
    assert across / ref == pytest.approx(
        path / THICKNESS, abs=max(5 * se_across / ref, 0.03)
    )


def test_the_same_seed_gives_the_same_regions(tmp_path):
    regions = doc(
        box("water", -FAR, 200.0), box("dense", 200.0, FIELD + FAR, material="dense")
    )
    a, _ = render(tmp_path / "a", regions, materials={"dense": DENSE})
    b, _ = render(tmp_path / "b", regions, materials={"dense": DENSE})
    c, _ = render(tmp_path / "c", regions, materials={"dense": DENSE}, seed=1235)
    same = np.abs(a - b).max() / np.abs(a).max()
    different = np.abs(a - c).max() / np.abs(a).max()
    print("same seed %.3g, different seed %.3g relative" % (same, different))
    assert same < 1e-4
    assert different > 0.1


def test_a_film_with_an_empty_hole(tmp_path):
    # A dense film with a vacuum cylinder through it, and plain vacuum beyond
    # x = 330 A: inside the hole the wave is the vacuum's
    if not engine_features(multem)["vacuum_regions"]:
        pytest.skip("this engine has no vacuum regions")
    hole, radius = (FIELD / 2, FIELD / 2), 80.0
    psi, _ = render(
        tmp_path / "film",
        doc(
            box("film", -FAR, 330.0, material="dense"),
            dict(
                id="hole",
                kind="cylinder",
                material="vacuum",
                parent="film",
                centre_A=[hole[0], hole[1], THICKNESS / 2],
                axis=[0, 0, 1],
                radius_A=radius,
                length_A=THICKNESS,
            ),
        ),
        materials={"dense": DENSE},
    )
    phase = phase_against(psi, window(345, 380, 0, FIELD))
    film, se_film = mean_and_se(phase[window(10, 90, 0, FIELD)])
    rows, cols = np.mgrid[0 : psi.shape[0], 0 : psi.shape[1]]
    x = (cols - MARGIN) * ANGPIX
    y = (rows - MARGIN) * ANGPIX
    inside = (x - hole[0]) ** 2 + (y - hole[1]) ** 2 < (radius - 20) ** 2
    in_hole, se_hole = masked_mean_and_se(phase, inside)
    film_sd = phase[window(10, 90, 0, FIELD)].std()
    print(
        "film %.4f +- %.4f rad (s.d. %.4f); hole %.5f +- %.5f rad (s.d. %.2g)"
        % (film, se_film, film_sd, in_hole, se_hole, phase[inside].std())
    )
    assert film > 10 * se_film
    assert abs(in_hole) < max(5 * se_hole, 0.01 * film)
    assert phase[inside].std() < 0.05 * film_sd


def test_plain_ice_takes_its_own_variance_per_molecule(tmp_path):
    # Plain ice (no regions) with twice water's variance per molecule: the
    # engine's single-ice path would keep water's, so Parakeet sends it through
    # the regions path, and the phase variance doubles (a weak phase object)
    base, _ = render(tmp_path / "water", None)
    doubled, _ = render_with_ice(
        tmp_path / "doubled", {"variance_constant": 2 * 10195.82}
    )
    whole = (slice(MARGIN, MARGIN + NX), slice(MARGIN, MARGIN + NX))
    a = np.angle(base[whole] / base[whole].mean()).var()
    b = np.angle(doubled[whole] / doubled[whole].mean()).var()
    print("phase variance: water %.5g, twice V0 %.5g, ratio %.4f" % (a, b, b / a))
    assert b / a == pytest.approx(2.0, rel=0.05)


def test_the_zero_loss_filter_follows_each_materials_path(tmp_path):
    # Water for x < 192 A and a material with half its inelastic mean free
    # path beyond, both 120 A thick in vacuum: behind the energy filter each
    # half keeps the fraction its own optical depth gives
    import parakeet.inelastic as inelastic

    mfp = inelastic.MEAN_FREE_PATH
    psi, path = render(
        tmp_path / "filtered",
        doc(
            box("water", -FAR, 192.0),
            box("dense", 192.0, FIELD + FAR, material="dense"),
        ),
        materials={"dense": dict(DENSE, inelastic_mfp_A=mfp / 2)},
    )
    cfg = yaml.safe_load(path.read_text())
    cfg["simulation"].update(inelastic_model="zero_loss", mp_loss_width=20)
    path.write_text(yaml.safe_dump(cfg))
    d = path.parent
    parakeet.simulate.optics(str(path), str(d / "exit_wave.h5"), str(d / "optics.h5"))
    image = np.asarray(parakeet.io.open(str(d / "optics.h5")).data[0], dtype=np.float64)

    # The image after the optics has no margin
    cols = np.arange(NX) * ANGPIX
    water = image[:, (cols > 40) & (cols < 150)].mean()
    dense = image[:, (cols > 234) & (cols < 344)].mean()
    from parakeet.config import Beam

    zlp = Beam().energy_spread * 300 * 1000.0  # the source's FWHM, eV
    expect_water = inelastic.filter_transmission_tau(
        THICKNESS / mfp, slit_width=20, zlp_fwhm=zlp
    )
    expect_dense = inelastic.filter_transmission_tau(
        2 * THICKNESS / mfp, slit_width=20, zlp_fwhm=zlp
    )
    print(
        "zero-loss means: water %.5f (expects %.5f x |psi|^2), dense %.5f (expects %.5f x |psi|^2)"
        % (water, expect_water, dense, expect_dense)
    )
    # The intensity before the filter is ~1 in both halves (a weak phase object)
    assert water == pytest.approx(expect_water, rel=0.01)
    assert dense == pytest.approx(expect_dense, rel=0.01)


def test_a_vacuum_region_needs_the_engine_to_support_it(tmp_path):
    if engine_features(multem)["vacuum_regions"]:
        pytest.skip("this engine supports vacuum regions")
    with pytest.raises(EngineFeatureError, match="vacuum_regions"):
        render(
            tmp_path / "hole",
            doc(
                box("film", -FAR, FIELD + FAR),
                dict(
                    id="hole",
                    kind="cylinder",
                    material="vacuum",
                    parent="film",
                    centre_A=[FIELD / 2, FIELD / 2, THICKNESS / 2],
                    axis=[0, 0, 1],
                    radius_A=50.0,
                    length_A=THICKNESS,
                ),
            ),
        )


def potential_of(directory, regions, materials=None, origin=(0, 0)):
    """The projected potential volume (slices, ny, nx) of render()'s sample"""
    _, path = render(directory, regions, materials=materials, origin=origin)
    parakeet.simulate.potential(
        str(path), str(directory / "sample.h5"), str(directory / "potential")
    )
    import mrcfile

    with mrcfile.open(str(directory / "potential_0.mrc")) as m:
        return np.asarray(m.data, dtype=np.float64)


def test_the_projected_potential_of_regions_needs_the_engine(tmp_path):
    if engine_features(multem)["multi_region_potential"]:
        pytest.skip("this engine has a multi-region projected potential")
    with pytest.raises(EngineFeatureError, match="multi_region_potential"):
        potential_of(tmp_path / "potential", doc(medium="water"))


def test_the_projected_potential_of_regions(tmp_path):
    # A dense square (twice water's molecules) in the middle of a water slab:
    # each slice's mean potential is the engine's own per-slice mean for its
    # material (multem.ice_slice_statistics), and nothing adds where they meet
    if not engine_features(multem)["multi_region_potential"]:
        pytest.skip("this engine has no multi-region projected potential")
    lo, hi = FIELD / 4, 3 * FIELD / 4
    volume = potential_of(
        tmp_path / "potential",
        doc(
            box("slab", -FAR, FIELD + FAR),
            box("square", lo, hi, material="dense", y0=lo, y1=hi, parent="slab"),
        ),
        materials={"dense": DENSE},
    )
    dz = 3.0  # the default slice thickness
    inside = slice(1, int(THICKNESS / dz) - 1)  # slices wholly inside the slab
    n = volume.shape[-1]
    ix = (np.arange(n) + 0.5) * ANGPIX

    def region(a, b):
        sel = (ix > a) & (ix < b)
        return np.ix_(sel, sel)

    centre = region(lo + 20, hi - 20)
    corner = region(0, lo - 20)
    expected = {
        name: multem.ice_slice_statistics(
            parakeet.simulate.engine.grf_parameters(m), ANGPIX, ANGPIX, dz
        )[0]
        for name, m in (("water", {"density": WATER_DENSITY}), ("dense", DENSE))
    }
    for name, window_ in (("water", corner), ("dense", centre)):
        per_slice = np.array([v[window_].mean() for v in volume[inside]])
        mean = per_slice.mean()
        se = per_slice.std(ddof=1) / np.sqrt(per_slice.size)
        print(
            "%s: slice mean %.5g +- %.2g, engine %.5g (%d slices)"
            % (name, mean, se, expected[name], per_slice.size)
        )
        assert mean == pytest.approx(expected[name], rel=0.01, abs=5 * se)


def test_the_potential_and_the_exit_wave_cover_the_same_field(tmp_path):
    # With the detector's origin moved, a dense band at sample x 300..400 A
    # falls on the field's x 100..200 A in the projected potential and in the
    # exit wave alike (and along x, not y, in both)
    if not engine_features(multem)["multi_region_potential"]:
        pytest.skip("this engine has no multi-region projected potential")
    origin = (200, 0)
    regions = doc(
        box("slab", -FAR, FIELD + FAR),
        box("band", 300.0, 400.0, material="dense", parent="slab"),
    )
    volume = potential_of(
        tmp_path / "potential", regions, materials={"dense": DENSE}, origin=origin
    )
    psi, _ = render(tmp_path / "wave", regions, materials={"dense": DENSE}, origin=origin)
    profiles = {
        "potential": volume.sum(axis=0).mean(axis=0),
        "potential across": volume.sum(axis=0).mean(axis=1),
        "exit wave": np.unwrap(np.angle(psi[MARGIN:-MARGIN, MARGIN:-MARGIN]))
        .mean(axis=0),
    }

    def edges(profile):
        low, high = np.percentile(profile, [10, 90])
        above = np.flatnonzero(profile > (low + high) / 2)
        return above.min(), above.max(), (high - low) / abs(high)

    expected = (100.0 / ANGPIX, 200.0 / ANGPIX)
    for name in ("potential", "exit wave"):
        first, last, contrast = edges(profiles[name])
        print("%s: band at pixels %d..%d, expected %.1f..%.1f" % ((name, first, last) + expected))
        assert contrast > 0.2
        assert abs(first - expected[0]) <= 1.5 and abs(last - expected[1]) <= 1.5
    across = profiles["potential across"]
    assert across.std() < 0.05 * abs(across.mean())
