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
    directory,
    regions,
    materials=None,
    seed=1234,
    ice_parameters=None,
    origin=(0, 0),
    simulation=None,
    scan=None,
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
        "scan": scan or {"mode": "still", "num_images": 1},
        "simulation": {
            "ice": True,
            "seed": seed,
            "margin": MARGIN,
            "padding": 20,
            "materials": materials,
            **({"ice_parameters": ice_parameters} if ice_parameters else {}),
            **(simulation or {}),
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


def elastic_out_images(directory, regions, materials, tilt, on=None, lens=None):
    """The same exit wave through the optics without and with elastic_out"""
    scan = (
        None
        if tilt == 0
        else {"mode": "tilt_series", "start_angle": tilt, "step_angle": 0, "num_images": 1}
    )
    _, path = render(directory, regions, materials=materials, scan=scan)
    d = path.parent
    images = {}
    for name, elastic_out in (("off", None), ("on", on or {"model": "iam"})):
        cfg = yaml.safe_load(path.read_text())
        cfg["simulation"]["elastic_out"] = elastic_out
        if lens is not None:
            cfg["microscope"]["lens"] = lens
        cfg["output"] = {"dtype": "float32"}  # float16 rounds at ~5e-4
        c = d / ("config_%s.yaml" % name)
        c.write_text(yaml.safe_dump(cfg))
        parakeet.simulate.optics(str(c), str(d / "exit_wave.h5"), str(d / ("%s.h5" % name)))
        images[name] = np.asarray(
            parakeet.io.open(str(d / ("%s.h5" % name))).data[0], dtype=np.float64
        )
    return images


@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_the_elastic_out_term_scales_a_uniform_slab_exactly(tmp_path, tilt):
    # Water filling the whole grid, 120 A thick: the term is one factor on the
    # whole exit wave, so every pixel of the image falls by exp(-L / lambda),
    # L = 120 A / cos tilt
    import parakeet.elastic as elastic

    images = elastic_out_images(
        tmp_path / "slab", doc(box("water", -FAR, FIELD + FAR)), None, tilt
    )
    ratio = images["on"] / images["off"]
    length = THICKNESS / np.cos(np.radians(tilt))
    water = elastic.water_number_density(WATER_DENSITY)
    expect = np.exp(-length * sum(elastic.missing_cross_sections(water, 300, ANGPIX)))
    print("uniform slab at %g deg: ratio %.8f +- %.1e, expects %.8f" % (tilt, ratio.mean(), ratio.std(), expect))
    assert ratio.mean() == pytest.approx(expect, rel=1e-6)
    assert ratio.std() < 1e-6


@pytest.mark.parametrize("tilt", [0.0, 30.0])
def test_the_elastic_out_term_removes_each_materials_share(tmp_path, tilt):
    # Water for x < 192 A and twice its atoms beyond: each half's mean falls by
    # its own exp(-L / lambda). Pixel by pixel the ratio is not uniform: the
    # step in the factor at x = 192 A (and at the grid's periodic wrap) is
    # delocalised by the defocus like any other edge, so means only
    import parakeet.elastic as elastic

    water = elastic.water_number_density(WATER_DENSITY)
    dense = {k: 2 * v for k, v in water.items()}
    images = elastic_out_images(
        tmp_path / "slab",
        doc(
            box("water", -FAR, 192.0),
            box("dense", 192.0, FIELD + FAR, material="dense"),
        ),
        {"dense": dict(DENSE, number_density_A3=dense)},
        tilt,
    )
    cols = np.arange(NX) * ANGPIX
    length = THICKNESS / np.cos(np.radians(tilt))
    for half, n, (x0, x1) in (("water", water, (40, 150)), ("dense", dense, (234, 344))):
        expect = np.exp(-length * sum(elastic.missing_cross_sections(n, 300, ANGPIX)))
        sel = (slice(20, -20), (cols > x0) & (cols < x1))
        got = images["on"][sel].mean() / images["off"][sel].mean()
        print("%s at %g deg: mean ratio %.7f, expects %.7f" % (half, tilt, got, expect))
        assert got == pytest.approx(expect, rel=2e-4)


def test_the_halo_puts_the_band_back_into_a_uniform_slab(tmp_path):
    # No aperture, no C_s and 2000 A of defocus: every ring is under ~300 A,
    # so in the middle of a slab filling the field the halo returns nearly
    # all that the term removed, and the image hardly falls. Without the
    # halo it falls by exp(-tau)
    import parakeet.elastic as elastic

    lens = {"c_10": -2000.0, "c_30": 0.0}
    images = elastic_out_images(
        tmp_path / "slab",
        doc(box("water", -FAR, FIELD + FAR)),
        None,
        0.0,
        on={"model": "iam", "halo": True},
        lens=lens,
    )
    water = elastic.water_number_density(WATER_DENSITY)
    removed = -np.expm1(-THICKNESS * sum(elastic.missing_cross_sections(water, 300, ANGPIX)))
    centre = images["on"][96:160, 96:160].mean() / images["off"][96:160, 96:160].mean()
    print("halo: centre ratio %.7f; removed %.5f" % (centre, removed))
    assert 1 - centre < 0.1 * removed  # over 90% comes back to the centre
    assert centre <= 1 + 1e-6  # and never more than was removed


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


def potential_of(directory, regions, materials=None, origin=(0, 0), **kwargs):
    """The projected potential volume (slices, ny, nx) of render()'s sample"""
    _, path = render(directory, regions, materials=materials, origin=origin, **kwargs)
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


def test_partial_volume_slices_give_a_thin_slab_its_thickness(tmp_path):
    # A 20 A slab at four offsets against the slice grid (which starts at the
    # anchor atom at z = 1 A). With partial-volume slice masks its projected
    # mean is 20 A of water at every offset, to the bound dz / (2 n) per face;
    # with the midpoint test it is a whole number of slices, off by up to dz
    if not engine_features(multem)["partial_volume"]:
        pytest.skip("this engine has no partial-volume slice masks")
    dz, n, thick = 3.0, 16, 20.0
    per_slice = multem.ice_slice_statistics(
        parakeet.simulate.engine.grf_parameters(
            {"density": WATER_DENSITY, "variance_constant": 1e-6}
        ),
        ANGPIX,
        ANGPIX,
        dz,
    )[0]
    expected = per_slice * thick / dz
    centre = slice(NX // 4, 3 * NX // 4)  # away from the anchors' corner
    errors = {}
    for samples in (n, 0):
        for z0 in (50.0, 50.75, 51.5, 52.25):
            volume = potential_of(
                tmp_path / ("n%d_z%.2f" % (samples, z0)),
                doc(box("thin", -FAR, FIELD + FAR, z0=z0, z1=z0 + thick)),
                ice_parameters={"variance_constant": 1e-6},
                simulation={"partial_volume_samples": samples},
            )
            measured = volume.sum(axis=0)[centre, centre].mean()
            errors[samples, z0] = (measured / expected - 1) * thick  # A of water
            print("n %2d, z0 %.2f: %+.3f A" % (samples, z0, errors[samples, z0]))
    for z0 in (50.0, 50.75, 51.5, 52.25):
        assert abs(errors[n, z0]) < 2 * dz / (2 * n) + 0.02
    assert max(abs(errors[0, z0]) for z0 in (50.0, 50.75, 51.5, 52.25)) > 0.5


CONVEX_CHILDREN = [
    dict(
        id="rotated_box",
        kind="box",
        # off the pixel lattice: at (192, 192) many pixel rays lie exactly in
        # the vertical faces (4i + 3j = 1096), where the engine's n.p <= d and
        # the box's half-open [-h, h) legitimately disagree
        centre_A=[192.37, 191.83, 60.0],
        half_lengths_A=[60.0, 30.0, 25.0],
        axes=[[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
    ),
    dict(
        id="cut_ellipsoid",
        kind="ellipsoid",
        centre_A=[190.0, 195.0, 58.0],
        semi_axes_A=[80.0, 50.0, 30.0],
        axes=[[0.8, 0.0, -0.6], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]],
        cuts=[{"normal": [1.0, 0.3, 0.0], "offset_A": 40.0}],
    ),
    dict(
        id="shell",
        kind="ellipsoidal_shell",
        centre_A=[192.0, 192.0, 60.0],
        semi_axes_A=[100.0, 70.0, 45.0],
        inner={"mode": "offset", "thickness_A": 12.0},
        cuts=[{"normal": [0.0, 1.0, 0.0], "offset_A": 50.0}],
        cut_offset_A=6.0,
    ),
    dict(
        id="tilted_slab",
        kind="slab",
        origin_A=[192.0, 192.0, 60.0],
        normal=[0.3, 0.0, 1.0],
        u_axis=[1.0, 0.0, 0.0],
        thickness_A=20.0,
        footprint={"kind": "rectangle", "half_u_A": 110.0, "half_v_A": 90.0,
                   "clips": [{"normal_uv": [1.0, -1.0], "offset_A": 60.0}]},
    ),
    dict(
        id="disc_slab",
        kind="slab",
        origin_A=[192.0, 192.0, 60.0],
        normal=[0.0, 0.4, 1.0],
        u_axis=[1.0, 0.0, 0.0],
        thickness_A=24.0,
        footprint={"kind": "disc", "centre_uv_A": [10.0, -5.0], "radius_A": 100.0},
    ),
]


@pytest.mark.parametrize("child", CONVEX_CHILDREN, ids=lambda c: c["id"])
def test_convex_regions_project_their_exact_paths(tmp_path, child):
    # A dense child of each kind the engine draws as a convex set, inside a
    # water slab, with partial-volume slices: the projected potential of every
    # pixel is sum over materials of (the engine's slice mean / dz) x the exact
    # path through that material (parakeet.inelastic.path_length_maps)
    if not engine_features(multem)["convex_regions"]:
        pytest.skip("this engine has no convex regions")
    import parakeet.inelastic
    import parakeet.sample._regions as R

    regions = doc(
        box("slab", -FAR, FIELD + FAR),
        dict(child, material="dense", parent="slab"),
    )
    still = {"variance_constant": 1e-6}
    volume = potential_of(
        tmp_path / child["id"],
        regions,
        materials={"dense": dict(DENSE, **still)},
        ice_parameters=still,
        simulation={"partial_volume_samples": 1},
    )
    dz = 3.0
    mean = {
        name: multem.ice_slice_statistics(
            parakeet.simulate.engine.grf_parameters(m), ANGPIX, ANGPIX, dz
        )[0]
        / dz
        for name, m in (
            ("water", {"density": WATER_DENSITY, **still}),
            ("dense", dict(DENSE, **still)),
        )
    }
    shape = {
        "type": "cuboid",
        "cuboid": {"length_x": FIELD, "length_y": FIELD, "length_z": THICKNESS},
    }
    L = parakeet.inelastic.path_length_maps(
        R.parse(regions),
        shape,
        (FIELD / 2, FIELD / 2, THICKNESS / 2),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0, 0),
        NX,
        NX,
        0,
        ANGPIX,
        stride=1,
        exact=True,
    )
    predicted = sum(mean[m] * L[m] for m in ("water", "dense") if m in L)
    measured = volume.sum(axis=0)
    # away from the anchor atoms' corner
    core = (slice(0, NX - 16), slice(0, NX - 16))
    error = np.abs(measured[core] - predicted[core])
    print(
        "%s: max |V - predicted| %.3g of %.3g (dense pixels %d)"
        % (child["id"], error.max(), predicted.max(), int((L["dense"][core] > 0).sum()))
    )
    assert (L["dense"][core] > 1).sum() > 1000
    assert error.max() < 1e-4 * predicted.max()


def test_a_clip_document_renders_owners_rule(tmp_path):
    # containment: clip with a cut shell and an ellipsoid poking through a
    # tilted slab: the engine clips them by the slab's planes, and the projected
    # potential of every pixel is the engine's slice mean x the exact path
    # through each material, where the path maps follow owner()
    if not engine_features(multem)["convex_regions"]:
        pytest.skip("this engine has no convex regions")
    import parakeet.inelastic
    import parakeet.sample._regions as R

    slab = dict(
        id="slab",
        kind="slab",
        material="water",
        origin_A=[192.37, 191.83, 60.0],
        normal=[0.1, 0.0, 1.0],
        u_axis=[1.0, 0.0, 0.0],
        thickness_A=50.0,
        footprint={"kind": "rectangle", "half_u_A": 2 * FAR, "half_v_A": 2 * FAR},
    )
    wall = dict(
        id="wall",
        kind="ellipsoidal_shell",
        material="dense",
        parent="slab",
        centre_A=[150.3, 190.7, 58.0],
        semi_axes_A=[110.0, 80.0, 70.0],
        inner={"mode": "offset", "thickness_A": 15.0},
        cuts=[{"normal": [1.0, 0.0, 0.0], "offset_A": 70.0}],
        cut_offset_A=5.0,
    )
    cell = dict(
        id="cell",
        kind="ellipsoid",
        material="dense",
        parent="slab",
        # apart from the wall: siblings that overlap are subtracted by their
        # partial-volume fractions, which is exact only along rays where one
        # lies inside the other
        centre_A=[320.6, 200.2, 61.0],
        semi_axes_A=[50.0, 90.0, 60.0],
    )
    regions = dict(doc(slab, wall, cell), containment="clip")
    still = {"variance_constant": 1e-6}
    volume = potential_of(
        tmp_path / "clip",
        regions,
        materials={"dense": dict(DENSE, **still)},
        ice_parameters=still,
        simulation={"partial_volume_samples": 1},
    )
    dz = 3.0
    mean = {
        name: multem.ice_slice_statistics(
            parakeet.simulate.engine.grf_parameters(m), ANGPIX, ANGPIX, dz
        )[0]
        / dz
        for name, m in (
            ("water", {"density": WATER_DENSITY, **still}),
            ("dense", dict(DENSE, **still)),
        )
    }
    shape = {
        "type": "cuboid",
        "cuboid": {"length_x": FIELD, "length_y": FIELD, "length_z": THICKNESS},
    }
    L = parakeet.inelastic.path_length_maps(
        R.parse(regions),
        shape,
        (FIELD / 2, FIELD / 2, THICKNESS / 2),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0, 0),
        NX,
        NX,
        0,
        ANGPIX,
        stride=1,
        exact=True,
    )
    predicted = sum(mean[m] * L[m] for m in ("water", "dense") if m in L)
    measured = volume.sum(axis=0)
    core = (slice(0, NX - 16), slice(0, NX - 16))
    error = np.abs(measured[core] - predicted[core])
    print("clip: max |V - predicted| %.3g of %.3g" % (error.max(), predicted.max()))
    # the clipped children are thinner than the slab everywhere: no pixel sees
    # more dense path than the slab's
    assert L["dense"][core].max() <= 50.0 / np.cos(np.arctan(0.1)) + 1e-6
    assert (L["dense"][core] > 1).sum() > 1000
    assert error.max() < 1e-4 * predicted.max()


# --- The masker's partition against owner(), pixel by pixel ---------------
#
# The engine paints owner()'s partition by subtraction: a region's slice mask
# is its shape less every shape that outranks it, and its partial-volume
# fraction is max(f_self - sum f_sub, 0) (python-multem compute_region_fraction).
# These build the masker through SimulationEngine.multi_region_masker, as a
# render does, and compare the engine's test hooks with owner() and with the
# exact owner() fraction of each pixel's column through the slice.

PN = 96  # test grid (px)
PDX = 1.0  # A per pixel: fine enough to resolve a band of a few pixels


def engine_partition(document, tilt_deg, monkeypatch, origin=(0.37, 0.29)):
    """The engine's masker for a document, at a tilt about y, on a PN x PN
    grid; with the document id of each masker slot and the rotation"""
    import types

    import parakeet.config
    import parakeet.sample._new
    from parakeet.simulate._exit_wave import resolve_ice_seed
    from scipy.spatial.transform import Rotation

    E = parakeet.simulate.engine
    slots = []
    real_region, real_sample = E.set_region_shape, E.set_sample_shape

    def region_shape(masker, region, place, clip_planes=None):
        slots.append(region.id)
        return real_region(masker, region, place, clip_planes)

    def sample_shape(masker, shape, centre):
        slots.append("<medium>")
        return real_sample(masker, shape, centre)

    monkeypatch.setattr(E, "set_region_shape", region_shape)
    monkeypatch.setattr(E, "set_sample_shape", sample_shape)
    centre = np.array([PN * PDX / 2, PN * PDX / 2, 40.0])
    shape = {"type": "cuboid", "cuboid": {"length_x": 4000.0, "length_y": 4000.0, "length_z": 80.0}}
    config = parakeet.config.load(
        {
            "sample": {"box": [PN * PDX, PN * PDX, 80.0], "centre": centre.tolist(), "shape": shape},
            "simulation": {
                "ice": True,
                "seed": 7,
                "partial_volume_samples": 4,
                "materials": {"dense": DENSE, "denser": {"density": 3 * WATER_DENSITY}},
            },
        }
    )
    simulation = resolve_ice_seed(config.simulation.model_dump())
    regions = parakeet.sample._new.regions_document(
        parakeet.config.SampleRegions(document=document)
    )
    sample = types.SimpleNamespace(regions=regions, centre=centre, shape=shape)
    scan = types.SimpleNamespace(image_number=[0], is_uniform_angular_scan=False)
    fake = types.SimpleNamespace(
        input=types.SimpleNamespace(nx=PN, ny=PN, spec_lx=PN * PDX, spec_ly=PN * PDX, spec_dz=3.0)
    )
    rot = Rotation.from_rotvec([0.0, np.radians(tilt_deg), 0.0])
    masker = E.SimulationEngine.multi_region_masker(
        fake, 0, PDX, origin, 0.0, rot.as_rotvec(), (0.0, 0.0, 0.0), sample, scan, simulation
    )
    assert masker.num_regions() == len(slots)
    # a sample point p sits at R (p - c) + c - (origin, 0) on the grid
    t = np.array([-origin[0], -origin[1], 0.0])
    return masker, slots, rot, centre, t, regions


def owner_partition(masker, slots, rot, centre, t, regions, zs, ze):
    """Per pixel: the engine's label and owner()'s at the slice mid-plane, and
    the engine's effective fraction per region and owner()'s exact fraction of
    the pixel's column, both (n_regions, PN * PN)"""
    import parakeet.sample._regions as R

    _, regs = R.parse(regions)
    index = {r.id: r.index for r in regs}
    slot_index = np.array([index.get(s, -1) for s in slots])
    inv = rot.inv()
    ii, jj = np.meshgrid(np.arange(PN), np.arange(PN), indexing="ij")  # [x, y]

    def sample_points(z):
        lab = np.stack([ii * PDX, jj * PDX, np.full(ii.shape, z)], -1).reshape(-1, 3)
        return inv.apply(lab - t - centre) + centre

    eng = np.array(multem.compute_region_labels_engine(masker, zs, ze, True))
    eng = np.where(eng >= 0, slot_index[np.clip(eng, 0, None)], eng)
    mid = sample_points(0.5 * (zs + ze))
    ref = R.owner(regs, mid)
    tie = np.zeros(len(ref), bool)
    for v in np.vstack([np.eye(3), -np.eye(3)]):
        tie |= R.owner(regs, mid + 1e-3 * PDX * v) != ref

    f_eng = np.zeros((len(regs), PN * PN))
    if hasattr(multem, "compute_region_fraction_engine"):
        # the engine's own composition, as the simulation runs it
        for s in range(len(slots)):
            if slot_index[s] >= 0:
                f_eng[slot_index[s]] = multem.compute_region_fraction_engine(masker, s, zs, ze, True)
    else:
        # an engine before it: the legacy composition, from each shape's fraction
        f_shape = [
            np.array(multem.compute_fraction_engine(masker.region(s).shape, zs, ze, 4, True))
            for s in range(len(slots))
        ]
        for s in range(len(slots)):
            f = f_shape[s]
            for u in masker.region(s).subtract_regions:
                f = np.maximum(f - f_shape[u], 0.0)
            if slot_index[s] >= 0:
                f_eng[slot_index[s]] = f
    p0, d, length = sample_points(zs), inv.apply([0.0, 0.0, 1.0]), ze - zs
    cuts = [np.zeros((len(p0), 1)), np.full((len(p0), 1), length)]
    for r in regs:
        c = R.boundary_crossings(r, p0, d)
        cuts.append(np.where(np.isfinite(c), np.clip(c, 0.0, length), length))
    cuts = np.sort(np.concatenate(cuts, axis=1), axis=1)
    mids, lens = 0.5 * (cuts[:, 1:] + cuts[:, :-1]), cuts[:, 1:] - cuts[:, :-1]
    own = R.owner(regs, (p0[:, None, :] + mids[..., None] * d).reshape(-1, 3)).reshape(mids.shape)
    f_ref = np.array([(lens * (own == r.index)).sum(axis=1) / length for r in regs])
    return eng, ref, tie, f_eng, f_ref, regs


def clearance(f_ref):
    """Pixels that, with their 8 neighbours, lie wholly in one region (or none)
    through the slice"""
    pure = np.all((f_ref == 0) | (f_ref == 1), axis=0).reshape(PN, PN)
    clear = pure.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            clear &= np.roll(np.roll(pure, di, 0), dj, 1)
    clear[[0, -1], :] = False
    clear[:, [0, -1]] = False
    return clear.ravel()


@pytest.mark.parametrize("tilt", [0.0, 40.0])
def test_overlapping_siblings_and_negative_priorities_partition_as_owner(tilt, monkeypatch):
    # Two sibling cut ellipsoids in a slab overlap in a band 12 A (12 px)
    # wide: the one at priority 0 must own the band, the one at -1 the rest.
    # A sphere at -2 inside the -1 sibling is outranked by its own parent and
    # owns nothing. A top-level blob at -2 straddles the slab's top face and
    # owns only what lies outside the slab (the slab, at -1, outranks it)
    if not engine_features(multem)["convex_regions"]:
        pytest.skip("this engine has no convex regions")
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    document = doc(
        box("slab", -FAR, PN + FAR, y0=-FAR, y1=PN + FAR, z0=10.0, z1=70.0, priority=-1),
        dict(id="low", kind="ellipsoid", material="dense", parent="slab", priority=-1,
             centre_A=[30.0, 48.0, 40.0], semi_axes_A=[40.0, 60.0, 25.0], axes=I3,
             cuts=[{"normal": [1.0, 0.0, 0.0], "offset_A": 24.0}]),     # x <= 54
        dict(id="high", kind="ellipsoid", material="denser", parent="slab", priority=0,
             centre_A=[66.0, 48.0, 40.0], semi_axes_A=[40.0, 60.0, 25.0], axes=I3,
             cuts=[{"normal": [-1.0, 0.0, 0.0], "offset_A": 24.0}]),    # x >= 42
        dict(id="inside_low", kind="sphere", material="denser", parent="low", priority=-2,
             centre_A=[25.0, 48.0, 40.0], radius_A=8.0),
        dict(id="surface", kind="sphere", material="dense", priority=-2,
             centre_A=[48.0, 30.0, 72.0], radius_A=10.0),
    )
    masker, slots, rot, centre, t, regions = engine_partition(document, tilt, monkeypatch)
    hard = double = clear_tested = 0
    owned_by = {}
    for z in (20.0, 31.0, 40.0, 52.0, 66.0, 71.0, 76.0):
        eng, ref, tie, f_eng, f_ref, regs = owner_partition(
            masker, slots, rot, centre, t, regions, z - 1.5, z + 1.5
        )
        hard += int(((eng != ref) & ~tie).sum())
        double += int(((eng == -2) & ~tie).sum())
        clear = clearance(f_ref)
        clear_tested += int(clear.sum())
        assert np.array_equal(f_eng[:, clear], f_ref[:, clear]), (
            "z %.0f: %d clearance pixels differ"
            % (z, int(np.any(f_eng[:, clear] != f_ref[:, clear], axis=0).sum()))
        )
        for r in regs:
            owned_by[r.id] = owned_by.get(r.id, 0) + int((ref == r.index).sum())
    assert hard == 0 and double == 0, (hard, double)
    assert clear_tested > 5000
    # the band went to the priority-0 sibling, the -2 child got nothing, and
    # the surface blob kept its part above the slab
    assert owned_by["inside_low"] == 0
    assert owned_by["high"] > 0 and owned_by["low"] > 0 and owned_by["surface"] > 0


def test_layered_siblings_have_exact_fractions_across_the_band(monkeypatch):
    # Where a column crosses only the band's faces (one sibling's face lying
    # inside the other), f_self - f_sub is exact: the fractions over all
    # regions sum to what owner() gives, pixel by pixel
    if not engine_features(multem)["convex_regions"]:
        pytest.skip("this engine has no convex regions")
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    document = doc(
        box("slab", -FAR, PN + FAR, y0=-FAR, y1=PN + FAR, z0=-200.0, z1=280.0, priority=-1),
        dict(id="low", kind="ellipsoid", material="dense", parent="slab", priority=-1,
             centre_A=[-60.0, 48.0, 40.0], semi_axes_A=[130.0, 400.0, 230.0], axes=I3,
             cuts=[{"normal": [1.0, 0.0, 0.0], "offset_A": 114.0}]),    # x <= 54
        dict(id="high", kind="ellipsoid", material="denser", parent="slab", priority=0,
             centre_A=[156.0, 48.0, 40.0], semi_axes_A=[130.0, 400.0, 230.0], axes=I3,
             cuts=[{"normal": [-1.0, 0.0, 0.0], "offset_A": 114.0}]),   # x >= 42
    )
    masker, slots, rot, centre, t, regions = engine_partition(document, 40.0, monkeypatch)
    tested = 0
    for z in (25.0, 40.0, 55.0):
        eng, ref, tie, f_eng, f_ref, regs = owner_partition(
            masker, slots, rot, centre, t, regions, z - 1.5, z + 1.5
        )
        edge = ~clearance(f_ref)
        dev = np.abs(f_eng.sum(axis=0) - f_ref.sum(axis=0))[edge]
        tested += int(edge.sum())
        assert dev.max() < 1e-6, "z %.0f: sum off by up to %.3g" % (z, dev.max())
    assert tested > 100


def needs_interval_composition():
    if not engine_features(multem)["convex_regions"] or not engine_features(multem)["interval_composition"]:
        pytest.skip("this engine has no interval composition of fractions")


@pytest.mark.parametrize("tilt", [0.0, 50.0])
def test_abutting_and_nested_regions_have_exact_fractions(tilt, monkeypatch):
    # Two cut cells that abut along a plane (band 0), each with an organelle
    # whose bilayer child is its outer skin, and surface ice outranked by
    # everything: through every slice the regions' fractions sum to what
    # owner() gives, pixel by pixel (the legacy composition took material at
    # the abutting plane and twice at each bilayer)
    needs_interval_composition()
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    document = doc(
        box("slab", -FAR, PN + FAR, y0=-FAR, y1=PN + FAR, z0=10.0, z1=70.0, priority=-1),
        dict(id="cell_a", kind="ellipsoid", material="dense", parent="slab",
             centre_A=[20.0, 48.0, 40.0], semi_axes_A=[60.0, 80.0, 28.0], axes=I3,
             cuts=[{"normal": [1.0, 0.0, 0.0], "offset_A": 28.37}]),     # x <= 48.37
        dict(id="cell_b", kind="ellipsoid", material="dense", parent="slab",
             centre_A=[76.0, 48.0, 40.0], semi_axes_A=[60.0, 80.0, 28.0], axes=I3,
             cuts=[{"normal": [-1.0, 0.0, 0.0], "offset_A": 27.63}]),    # x >= 48.37
        dict(id="organelle", kind="ellipsoid", material="denser", parent="cell_a",
             centre_A=[30.0, 40.0, 41.0], semi_axes_A=[12.0, 14.0, 10.0], axes=I3),
        dict(id="bilayer", kind="ellipsoidal_shell", material="dense", parent="organelle",
             centre_A=[30.0, 40.0, 41.0], semi_axes_A=[12.0, 14.0, 10.0], axes=I3,
             inner={"mode": "offset", "thickness_A": 4.0}),
        dict(id="ice", kind="sphere", material="dense", priority=-2,
             centre_A=[60.0, 60.0, 74.0], radius_A=9.0),
    )
    masker, slots, rot, centre, t, regions = engine_partition(document, tilt, monkeypatch)
    worst, tested = 0.0, 0
    for z in np.arange(14.0, 84.0, 3.0):
        eng, ref, tie, f_eng, f_ref, regs = owner_partition(
            masker, slots, rot, centre, t, regions, z - 1.5, z + 1.5
        )
        edge = ~clearance(f_ref)
        tested += int(edge.sum())
        worst = max(worst, float(np.abs(f_eng - f_ref)[:, edge].max()) if edge.any() else 0.0)
    assert tested > 1000
    assert worst < 1e-6, worst


def test_pruning_the_paint_plan_leaves_the_labels_unchanged(monkeypatch):
    # The pruned plan subtracts a child only through its parent: the per-pixel
    # owner on every slice is the same, bit for bit
    needs_interval_composition()
    import parakeet.sample._regions as R

    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    document = doc(
        box("slab", -FAR, PN + FAR, y0=-FAR, y1=PN + FAR, z0=10.0, z1=70.0, priority=-1),
        dict(id="cell", kind="ellipsoid", material="dense", parent="slab",
             centre_A=[48.0, 48.0, 40.0], semi_axes_A=[45.0, 45.0, 28.0], axes=I3),
        dict(id="organelle", kind="ellipsoid", material="denser", parent="cell",
             centre_A=[40.0, 50.0, 41.0], semi_axes_A=[16.0, 14.0, 12.0], axes=I3),
        dict(id="bilayer", kind="ellipsoidal_shell", material="dense", parent="organelle",
             centre_A=[40.0, 50.0, 41.0], semi_axes_A=[16.0, 14.0, 12.0], axes=I3,
             inner={"mode": "offset", "thickness_A": 4.0}),
        dict(id="ice", kind="sphere", material="dense", priority=-2,
             centre_A=[48.0, 48.0, 72.0], radius_A=10.0),
    )
    labels = {}
    for prune in (True, False):
        real = R.paint_plan
        monkeypatch.setattr(R, "paint_plan", lambda regions, prune=prune, _real=real: _real(regions, prune))
        masker, *_ = engine_partition(document, 30.0, monkeypatch)
        labels[prune] = [
            np.array(multem.compute_region_labels_engine(masker, z - 1.5, z + 1.5, True))
            for z in np.arange(8.0, 86.0, 3.0)
        ]
        monkeypatch.setattr(R, "paint_plan", real)
    assert all(np.array_equal(a, b) for a, b in zip(labels[True], labels[False]))
