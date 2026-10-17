import shutil

import pytest

import check_shaders

pytestmark = pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="glslangValidator not installed")


@pytest.mark.parametrize("path", check_shaders.targets([]), ids=lambda p: str(p.relative_to(check_shaders.ROOT)))
def test_shader_compiles(path):
    ok, log = check_shaders.check(path, shutil.which("glslangValidator"))
    assert ok, log


def test_checker_reports_errors(tmp_path):
    bad = tmp_path / "bad.glsl"
    bad.write_text("#include <bus_glsl>\nout vec4 fragColor;\nvoid main() { fragColor = busMissing(vUV.st); }\n")
    ok, log = check_shaders.check(bad, shutil.which("glslangValidator"))
    assert not ok and "busMissing" in log


def test_compute_3d_stage_detection():
    stage, preamble = check_shaders.stage_of("layout (local_size_x = 4, local_size_y = 4, local_size_z = 4) in;")
    assert stage == "comp" and "image3D" in preamble
    stage, preamble = check_shaders.stage_of("layout (local_size_x = 8, local_size_y = 8) in;")
    assert stage == "comp" and "image2D" in preamble
    assert check_shaders.stage_of("void main(){}")[0] == "frag"
