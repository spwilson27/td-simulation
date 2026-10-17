import cache


def test_atlas_exact_for_demo_grids():
    for res in ([128, 128, 128], [96, 96, 96], [192, 192, 192], [256, 256, 256]):
        layout = cache.atlas_layout(res)
        assert layout["cols"] * layout["rows"] >= res[2]
        assert layout["width"] == layout["cols"] * res[0]
        assert layout["height"] == layout["rows"] * res[1]
        assert layout["width"] >= cache.MIN_ATLAS_EDGE
        assert layout["height"] >= cache.MIN_ATLAS_EDGE
        assert layout["width"] * layout["height"] == res[0] * res[1] * res[2]


def test_atlas_pads_small_volumes_for_movie_file_in():
    layout = cache.atlas_layout([32, 32, 32])
    assert layout["width"] >= cache.MIN_ATLAS_EDGE
    assert layout["height"] >= cache.MIN_ATLAS_EDGE
    assert layout["width"] * layout["height"] > 32 ** 3
    assert cache.estimate_bytes([32, 32, 32], {"d": "mono16float"}, 2) == layout["width"] * layout["height"] * 2 * 2


def test_atlas_pixel_roundtrip():
    layout = cache.atlas_layout([96, 96, 96])
    for z in (0, 1, 17, 95):
        for x, y in ((0, 0), (1, 2), (95, 95), (10, 40)):
            px, py = cache.atlas_pixel(x, y, z, layout)
            assert cache.atlas_voxel(px, py, layout) == (x, y, z)
    # A 128^3 atlas is exact, so every pixel is a voxel.
    layout = cache.atlas_layout([128, 128, 128])
    assert cache.atlas_voxel(0, 0, layout) == (0, 0, 0)
    assert cache.atlas_voxel(layout["width"] - 1, layout["height"] - 1, layout)[2] == 127
    assert cache.atlas_voxel(-1, 0, layout) is None
    assert cache.atlas_voxel(layout["width"], 0, layout) is None


def test_atlas_padding_pixels_are_not_voxels():
    layout = cache.atlas_layout([32, 32, 8])
    missing = 0
    for py in range(layout["height"]):
        for px in range(layout["width"]):
            if cache.atlas_voxel(px, py, layout) is None:
                missing += 1
    assert missing == layout["width"] * layout["height"] - 32 * 32 * 8
