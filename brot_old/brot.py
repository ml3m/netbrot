"""
3D Mandelbrot — True Attractor View (PyVista, OpenGL)
=====================================================
Clean bifurcation diagram: single Im(c)=0 slice so no stacking artefacts.
Re(z) on vertical axis gives the period-doubling / chaos structure.

Install:
    pip install pyvista numpy
Run:
    python mandelbrot_3d.py
"""
import numpy as np
import pyvista as pv


# ── bifurcation slice: Im(c) = 0 exactly ──────────────────────────────────────
def sample_bifurcation(
    re_min=-2.5,
    re_max=0.55,
    nx=7000,            # dense horizontal sampling
    warmup=800,         # long warmup so transients fully die out
    keep=2000,           # record many settled steps to fill chaotic region
    escape_r=2.0,
):
    """
    True bifurcation diagram: c is real only (Im=0).
    Each c value contributes up to `keep` (Re(c), Re(z_n)) points.
    No Im stacking → no opacity accumulation artefacts.
    """
    re = np.linspace(re_min, re_max, nx)
    C  = re.astype(complex)           # Im(c) = 0
    Z  = np.zeros(nx, dtype=complex)
    bounded = np.ones(nx, dtype=bool)

    print(f"  Bifurcation warmup ({warmup} iters) …", end=" ", flush=True)
    for _ in range(warmup):
        Z[bounded] = Z[bounded] ** 2 + C[bounded]
        bounded[np.abs(Z) > escape_r] = False
    print(f"{bounded.sum():,} bounded c values.")

    all_pts   = []
    all_color = []
    print(f"  Recording {keep} orbit steps …", end=" ", flush=True)
    for _ in range(keep):
        Z[bounded] = Z[bounded] ** 2 + C[bounded]
        bounded[np.abs(Z) > escape_r] = False
        rx = C[bounded].real
        rz = Z[bounded].real          # Re(z_n) — vertical axis
        # colour by Re(c): left=chaotic(purple/blue), right=ordered(yellow/white)
        col = rx
        # Im(c) = 0 for all; give slight jitter in y so points aren't co-planar
        iy = np.zeros_like(rx)
        all_pts.append(np.column_stack([rx, iy, rz]))
        all_color.append(col)

    points = np.vstack(all_pts).astype(np.float32)
    colors = np.concatenate(all_color).astype(np.float32)
    print(f"{len(points):,} bifurcation points.")
    return points, colors


# ── wide 3D attractor for structural context ───────────────────────────────────
def sample_orbits_wide(
    re_min=-2.5,
    re_max=0.55,
    im_min=-1.15,
    im_max=1.15,
    nx=900,
    ny=700,
    warmup=120,
    keep=80,
    escape_r=2.0,
):
    re = np.linspace(re_min, re_max, nx)
    im = np.linspace(im_min, im_max, ny)
    RE, IM = np.meshgrid(re, im)
    C = RE + 1j * IM
    Z = np.zeros_like(C)
    bounded = np.ones(C.shape, dtype=bool)

    print(f"  Wide warmup ({warmup} iters) …", end=" ", flush=True)
    for _ in range(warmup):
        Z[bounded] = Z[bounded] ** 2 + C[bounded]
        bounded[np.abs(Z) > escape_r] = False
    print(f"{bounded.sum():,} bounded.")

    all_pts, all_color = [], []
    print(f"  Wide recording {keep} steps …", end=" ", flush=True)
    for _ in range(keep):
        Z[bounded] = Z[bounded] ** 2 + C[bounded]
        bounded[np.abs(Z) > escape_r] = False
        rx = RE[bounded].ravel()
        iy = IM[bounded].ravel()
        rz = Z[bounded].real.ravel()
        col = np.angle(C[bounded].ravel())
        all_pts.append(np.column_stack([rx, iy, rz]))
        all_color.append(col)

    points = np.vstack(all_pts).astype(np.float32)
    colors = np.concatenate(all_color).astype(np.float32)
    print(f"{len(points):,} wide orbit points.")
    return points, colors


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Sampling BIFURCATION diagram (Im(c)=0 slice) …")
    pts_bif, col_bif = sample_bifurcation()

    print("\nSampling WIDE attractor (ghost skeleton) …")
    pts_wide, col_wide = sample_orbits_wide()

    cloud_bif  = pv.PolyData(pts_bif)
    cloud_bif["re_c"] = col_bif

    cloud_wide = pv.PolyData(pts_wide)
    cloud_wide["phase_c"] = col_wide

    pl = pv.Plotter(window_size=(1600, 960))
    pl.set_background("#050505")

    # Ghost wide attractor — very faint, just for 3D context
    pl.add_points(
        cloud_wide,
        scalars="phase_c",
        cmap="cool",
        clim=[-np.pi, np.pi],
        point_size=0.8,
        render_points_as_spheres=False,
        opacity=1.0,
        show_scalar_bar=False,
    )

    # Bifurcation diagram — crisp dots, low opacity so overlaps stay honest
    pl.add_points(
        cloud_bif,
        scalars="re_c",
        cmap="plasma",
        clim=[-2.5, 0.55],
        point_size=0.8,
        render_points_as_spheres=False,
        opacity=0.12,           # low: density encodes naturally via accumulation
        show_scalar_bar=False,
    )

    pl.add_axes(xlabel="Re(c)", ylabel="Im(c)", zlabel="Re(zₙ)", color="tomato")
    pl.add_text(
        "Mandelbrot — 3D Bifurcation Diagram\n"
        "Im(c) = 0  ·  Re(zₙ) vertical  ·  chaos at left",
        position="upper_left", font_size=10, color="tomato",
    )
    pl.add_text(
        "Left-drag: rotate   Right-drag: zoom   R: reset   Q: quit",
        position="lower_edge", font_size=8, color="#441100",
    )

    # Face the bifurcation tree front-on, slightly elevated
    pl.camera_position = [
        (-1.0, -5.0, 1.5),    # camera
        (-1.0,  0.0, 0.0),    # focal point — chaotic centre
        ( 0.0,  0.0, 1.0),    # up
    ]
    pl.camera.zoom(1.3)
    pl.show()


if __name__ == "__main__":
    main()
