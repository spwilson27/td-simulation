"""The reference projection has to match the shader: adjoint divergence, and a V-cycle that
actually removes it. These are small grids; the GPU check is validate_phase4."""

import numpy as np

import pressure3d as pressure


def _smooth_velocity(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    vel = np.zeros((n, n, n, 3))
    vel[..., 0] = 0.15 * np.sin(2 * np.pi * xx) * np.cos(np.pi * yy)
    vel[..., 1] = 0.10 * np.cos(np.pi * xx) * np.sin(2 * np.pi * yy)
    vel[..., 2] = 0.04 * np.sin(np.pi * zz)
    vel += 0.005 * rng.normal(size=vel.shape)
    return vel


def _solid(n):
    solid = np.zeros((n, n, n), dtype=bool)
    solid[n // 3:n // 3 + max(3, n // 7), n // 3:n // 2, n // 4:n // 2] = True
    return solid


def test_divergence_is_the_laplacian_of_the_gradient():
    rng = np.random.default_rng(1)
    pressure_field = rng.normal(size=(12, 12, 12))
    solid = _solid(12)
    pressure_field[solid] = 0.0
    got = pressure.divergence(pressure.gradient(pressure_field, solid=solid), solid=solid)
    want = pressure.laplacian(pressure_field, solid=solid)
    assert np.max(np.abs(got - want)) == 0.0


def test_vcycle_reduces_smooth_divergence_tenfold():
    vel = _smooth_velocity(32)
    solid = _solid(32)
    obstacle = np.zeros_like(vel)
    obstacle[..., 1] = 0.2
    rhs = pressure.remove_mean(pressure.divergence(vel, solid=solid, obstacle_vel=obstacle), solid)
    before = pressure.rms(pressure.divergence(vel, solid=solid, obstacle_vel=obstacle), solid)
    jacobi = pressure.jacobi(np.zeros_like(rhs), rhs, solid=solid, iterations=12, omega=1.0)
    after_jacobi = pressure.rms(
        pressure.divergence(pressure.project(vel, jacobi, solid=solid, obstacle_vel=obstacle),
                            solid=solid, obstacle_vel=obstacle),
        solid,
    )
    solved = pressure.solve(rhs, solid=solid, cycles=2, smooth=4, coarse_iters=32)
    after = pressure.rms(
        pressure.divergence(pressure.project(vel, solved, solid=solid, obstacle_vel=obstacle),
                            solid=solid, obstacle_vel=obstacle),
        solid,
    )
    assert before / after >= 10.0
    assert before / after > before / after_jacobi
