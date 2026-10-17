import json

import pytest

import cache
from cache import Manifest


def make_manifest(**over):
    kw = dict(sim="fluid2d", params={"res": [64, 64], "vorticity": 0.3}, fields={"vel": "rg32float", "scalars": "rgba16float"},
              res=[64, 64], bounds=[0, 0, 1, 1], fps=30, substeps=2, frame_start=1, frame_end=4)
    kw.update(over)
    return Manifest(**kw)


def touch_frame(root, m, frame, fields=None):
    for f in fields or m.fields:
        p = cache.frame_path(root, f, frame)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_param_hash_is_order_and_noise_insensitive():
    a = cache.param_hash({"a": 1, "b": 0.1 + 0.2, "c": [1, 2]})
    b = cache.param_hash({"c": [1, 2], "b": 0.3, "a": 1})
    assert a == b and len(a) == 16
    assert cache.param_hash({"a": 1, "b": 0.31}) != cache.param_hash({"a": 1, "b": 0.3})


def test_manifest_roundtrip_and_dt(tmp_path):
    m = make_manifest(extra={"note": "hi"})
    cache.write_manifest(tmp_path, m)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["dt"] == pytest.approx(1 / 60)
    back = cache.read_manifest(tmp_path)
    assert back == m and back.frame_count == 4
    assert not list(tmp_path.glob(".manifest.*"))


def test_read_manifest_tolerates_garbage(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json")
    assert cache.read_manifest(tmp_path) is None


def test_frame_path_layout(tmp_path):
    assert cache.frame_path(tmp_path, "vel", 12) == tmp_path / "vel" / "vel.00012.exr"


def test_status_and_resume(tmp_path):
    m = make_manifest()
    assert cache.cache_status(tmp_path, m.params) == cache.MISSING
    cache.write_manifest(tmp_path, m)
    assert cache.cache_status(tmp_path, m.params) == cache.PARTIAL
    assert cache.resume_frame(tmp_path, m) == 1
    touch_frame(tmp_path, m, 1)
    touch_frame(tmp_path, m, 2)
    touch_frame(tmp_path, m, 3, fields=["vel"])
    assert cache.resume_frame(tmp_path, m) == 3
    touch_frame(tmp_path, m, 3)
    touch_frame(tmp_path, m, 4)
    assert cache.resume_frame(tmp_path, m) == 5
    assert cache.cache_status(tmp_path, m.params) == cache.COMPLETE
    assert cache.cache_status(tmp_path, {**m.params, "vorticity": 0.5}) == cache.STALE
    assert cache.cache_status(tmp_path, m.params, frame_end=10) == cache.PARTIAL
    assert cache.cache_status(tmp_path, m.params, frame_start=0) == cache.STALE
    assert cache.cache_status(tmp_path, m.params, frame_start=2) == cache.STALE


def test_version_change_is_stale(tmp_path):
    m = make_manifest(version=0)
    cache.write_manifest(tmp_path, m)
    assert cache.cache_status(tmp_path, m.params) == cache.STALE


def test_clear_cache_only_removes_its_files(tmp_path):
    m = make_manifest()
    cache.write_manifest(tmp_path, m)
    touch_frame(tmp_path, m, 1)
    (tmp_path / "vel" / "notes.txt").write_text("keep")
    assert cache.clear_cache(tmp_path, m) == 3
    assert (tmp_path / "vel" / "notes.txt").exists()
    assert not (tmp_path / "scalars").exists()
    assert cache.read_manifest(tmp_path) is None


def test_estimate_bytes():
    assert cache.estimate_bytes([256, 256], {"vel": "rg32float", "s": "rgba16float"}, 10) == 256 * 256 * 16 * 10
    # 128^3 packs into a 16x8 tile grid with no padding, so the atlas is the voxel count.
    assert cache.estimate_bytes([128, 128, 128], {"d": "mono16float"}, 1) == 128 ** 3 * 2


@pytest.mark.parametrize("t,expected", [
    (2.0, (2, 3, 0.0)), (2.25, (2, 3, 0.25)), (0.0, (1, 2, 0.0)), (9.0, (4, 4, 0.0)), (3.999, (3, 4, 0.999)),
])
def test_blend_frames(t, expected):
    a, b, w = cache.blend_frames(t, 1, 4)
    assert (a, b) == expected[:2] and w == pytest.approx(expected[2])


def test_glsl_uniform_identity_skips_clock_and_keeps_expressions():
    blocks = [
        ("uClock", [("", 0.016), ("", 1.5), ("", 3.0), ("", 4.0)]),
        ("uSplatAmount", [("parent().par.Fuel", 6.0), ("", 2.0), ("", 0.0), ("", 0.0)]),
        ("uClock2", [("op.simclock.par.Seed", 0.0), ("", 0.0), ("", 0.0), ("", 0.0)]),
    ]
    assert cache.glsl_uniform_identity(blocks) == [["uSplatAmount", ["parent().par.Fuel", 2.0, 0.0, 0.0]]]
    changed = [("uSplatAmount", [("parent().par.Fuel", 9.0), ("", 2.0), ("", 0.0), ("", 0.0)])]
    assert cache.glsl_uniform_identity(changed) == cache.glsl_uniform_identity(blocks)


def test_custom_par_identity_uses_values_and_skips_pulses():
    pars = [("Fuel", "", 6.0, "Float"), ("Bake", "", 0, "Pulse"), ("Discspeed", "op.simclock.par.Time", 0.7, "Float")]
    assert cache.custom_par_identity(pars) == {"Fuel": 6.0, "Discspeed": "op.simclock.par.Time"}
    assert cache.custom_par_identity([("Fuel", "", 7.0, "Float")]) != cache.custom_par_identity([("Fuel", "", 6.0, "Float")])


def test_plan_bake_extends_resumes_and_stops_early():
    m = make_manifest()
    present = {1, 2, 3, 4}
    extended = cache.plan_bake(m, m.params, 1, 8, present)
    assert extended == {"action": "resume", "frame_start": 1, "frame_end": 8, "resume_frame": 5}
    hole = cache.plan_bake(m, m.params, 1, 8, {1, 2, 4})
    assert hole["action"] == "resume" and hole["resume_frame"] == 3 and hole["frame_end"] == 8
    shorter = cache.plan_bake(m, m.params, 1, 2, present)
    assert shorter["action"] == "complete" and shorter["frame_end"] == 2
    short_hole = cache.plan_bake(m, m.params, 1, 3, {1, 3, 4})
    assert short_hole == {"action": "resume", "frame_start": 1, "frame_end": 3, "resume_frame": 2}
    assert cache.plan_bake(m, m.params, 2, 4, present)["action"] == "stale"
    assert cache.plan_bake(m, {**m.params, "vorticity": 1}, 1, 4, present)["action"] == "stale"
    assert cache.plan_bake(make_manifest(version=0), m.params, 1, 4, present)["action"] == "stale"
    assert cache.plan_bake(None, m.params, 1, 4, set())["action"] == "missing"


def test_directory_plan_extension_keeps_existing_frames(tmp_path):
    m = make_manifest()
    cache.write_manifest(tmp_path, m)
    for frame in range(1, 5):
        touch_frame(tmp_path, m, frame)
    before = (tmp_path / "vel" / "vel.00001.exr").read_bytes()
    plan = cache.directory_plan(tmp_path, m.params, 1, 6)
    assert plan["action"] == "resume" and plan["resume_frame"] == 5 and plan["frame_end"] == 6
    assert cache.cache_status(tmp_path, m.params, frame_end=6) == cache.PARTIAL
    assert cache.cache_status(tmp_path, m.params, frame_end=2) == cache.COMPLETE
    assert (tmp_path / "vel" / "vel.00001.exr").read_bytes() == before


def test_clear_cache_without_manifest(tmp_path):
    m = make_manifest()
    touch_frame(tmp_path, m, 1)
    touch_frame(tmp_path, m, 2)
    (tmp_path / "vel" / "notes.txt").write_text("keep")
    assert cache.read_manifest(tmp_path) is None
    assert cache.clear_cache(tmp_path) == 4
    assert (tmp_path / "vel" / "notes.txt").exists()
    assert not list((tmp_path / "vel").glob("*.exr"))
    assert not (tmp_path / "scalars").exists()
    assert cache.clear_cache(tmp_path / "missing") == 0


def test_group_cache_paths_resume_and_staleness(tmp_path):
    members = {
        "fluid2d": {"params": {"drag": 1}, "fields": {"vel": "rg32float"}, "res": [8, 8], "bounds": [0, 0, 1, 1], "extra": {}},
        "particles": {"params": {"drag": 2}, "fields": {"pos": "rgba32float"}, "res": [4, 4], "bounds": [0, 0, 1, 1], "extra": {}},
    }
    params = {name: info["params"] for name, info in members.items()}
    m = make_manifest(sim="group", params=params, fields={}, members=members, frame_end=2)
    assert cache.frame_path(tmp_path, "vel", 1, "fluid2d") == tmp_path / "fluid2d" / "vel" / "vel.00001.exr"
    cache.write_manifest(tmp_path, m)
    for sim, info in members.items():
        for frame in (1, 2):
            for name in info["fields"]:
                path = cache.frame_path(tmp_path, name, frame, sim)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"a")
    assert cache.resume_frame(tmp_path, m) == 3
    assert cache.cache_status(tmp_path, params) == cache.COMPLETE
    changed = {"fluid2d": params["fluid2d"], "particles": {"drag": 9}}
    assert cache.cache_status(tmp_path, changed) == cache.STALE
    assert cache.clear_cache(tmp_path, m) >= 5
    assert not (tmp_path / "fluid2d").exists()
    assert cache.read_manifest(tmp_path) is None
