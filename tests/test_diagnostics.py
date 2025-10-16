import numpy as np
from interpolation.splines import UCGrid
from numbacs.flows import get_callable_scalar_linear
from numbacs.diagnostics import (
    ftle_grid_2D,
    C_tensor_2D,
    C_eig_aux_2D,
    C_eig_2D,
    lavd_grid_2D,
    ile_2D_func,
    S_eig_2D_func,
    S_2D_func,
    ile_2D_data,
    S_eig_2D_data,
    ivd_grid_2D
)

def reconstruct_matrix(eigvals, eigvecs):
    """
    Reconstruct matrix from spectral decomposition. Used to check that eigenvales/vectors
    are accurate since different linalg backends can lead to large enough differences in
    eigenvectors to fail tests.

    Parameters
    ----------
    eigvals : np.ndarray, shape = (nx, ny, 2)
        array of eigenvalues over grid.
    eigvecs : np.ndarray, shape = (nx, ny, 2, 2)
        array of eigenvectors over grid.

    Returns
    -------
    Ar: np.ndarray, shape = (nx, ny, 2, 2)
        array containing reconstructed matrices over grid.

    """
    eigvals_diag = np.zeros_like(eigvecs)
    eigvals_diag[: ,: ,0, 0] = eigvals[:, :, 0]
    eigvals_diag[:, :, 1, 1] = eigvals[:, :, 1]
    Ar = np.einsum(
        "...ij,...jk,...kl->...il",
        eigvecs,
        eigvals_diag,
        np.transpose(eigvecs, (0, 1, 3, 2)),
        out=np.zeros_like(eigvecs)
    )
    return Ar

def evecs_allclose(evecs, evecs_expected, rtol=1e-5, atol=1e-8):
    """
    Check if eigenvector angles are close
    """

    norms = np.linalg.norm(evecs, axis=-2)
    norms_expected = np.linalg.norm(evecs_expected, axis=-2)
    zero_mask = np.logical_and(norms < atol, norms_expected < atol)

    norms = np.expand_dims(norms, axis=2)
    norms_expected = np.expand_dims(norms_expected, axis=2)

    evecs_n = np.divide(evecs, norms, out=np.zeros_like(evecs), where=norms != 0)
    evecs_n_expected = np.divide(
        evecs_expected, norms_expected, out=np.zeros_like(evecs), where=norms_expected != 0
    )

    dotprod = np.einsum("...ab,...ab->...b", evecs_n, evecs_n_expected)
    parallel = np.isclose(np.abs(dotprod), 1.0, rtol=rtol, atol=atol)

    if not np.all(parallel):
        return False, dotprod, zero_mask
    else:
        return True, np.all(parallel | zero_mask)

def apply_mask(arr, mask):
    """Apply a mask to arr and return 0.0 where mask is True"""

    arr_masked = arr.copy()
    arr_masked[mask] = 0.0

    return arr_masked

def test_ftle_grid_2D(coords_dg, fm_data, ftle_data, mask_dg):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]
    T = 8.
    ftle = ftle_grid_2D(fm_data, T, dx, dy).astype(np.float32)

    assert np.allclose(ftle, ftle_data)

    ftle_masked = ftle_grid_2D(
        fm_data, T, dx, dy, mask=mask_dg, dilate_mask=False
    ).astype(np.float32)

    assert np.allclose(ftle_masked, apply_mask(ftle_data, mask_dg))


def test_C_tensor_2D(coords_dg, fm_aux_data, C_data, mask_dg):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]
    C = C_tensor_2D(fm_aux_data, dx, dy).astype(np.float32)

    assert np.allclose(C, C_data)

    C_masked = C_tensor_2D(fm_aux_data, dx, dy, mask=mask_dg, dilate_mask=False).astype(np.float32)

    assert np.allclose(C_masked, apply_mask(C_data, mask_dg))

def test_C_eig_aux_2D(coords_dg, fm_aux_data, C_eig_aux_data, mask_dg):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]
    Cvals_expected, Cvecs_expected = C_eig_aux_data
    Cvals, Cvecs = C_eig_aux_2D(fm_aux_data, dx, dy)
    C_expected = reconstruct_matrix(Cvals_expected, Cvecs_expected)
    C = reconstruct_matrix(Cvals.astype(np.float32), Cvecs.astype(np.float32))

    assert np.allclose(C, C_expected)

    Cvals_masked, Cvecs_masked = C_eig_aux_2D(
        fm_aux_data, dx, dy, mask=mask_dg, dilate_mask=False
    )
    C_masked = reconstruct_matrix(
        Cvals_masked.astype(np.float32), Cvecs_masked.astype(np.float32)
    )

    assert np.allclose(C_masked, apply_mask(C_expected, mask_dg))

def test_C_eig_2D(coords_dg, fm_data, C_eig_data, mask_dg):

    x, y = coords_dg
    dx = x[1]
    dy = y[1]
    Cvals_expected, Cvecs_expected = C_eig_data
    Cvals, Cvecs = C_eig_2D(fm_data, dx, dy)
    C_expected = reconstruct_matrix(Cvals_expected, Cvecs_expected)
    C = reconstruct_matrix(Cvals.astype(np.float32), Cvecs.astype(np.float32))

    assert np.allclose(C, C_expected)

    Cvals_masked, Cvecs_masked = C_eig_2D(fm_data, dx, dy, mask_dg, dilate_mask=False)
    C_masked = reconstruct_matrix(
        Cvals_masked.astype(np.float32), Cvecs_masked.astype(np.float32)
    )

    assert np.allclose(C_masked, apply_mask(C_expected, mask_dg))


def test_lavd_grid_2D(coords_dg, fm_n_data, vort_data, lavd_data, mask_dg):

    x, y = coords_dg
    [X, Y] = np.meshgrid(x, y, indexing='ij')
    xrav = X.ravel()
    yrav = Y.ravel()
    T = 8.
    tspan = np.linspace(0., 8., 4)
    grid_vort = UCGrid((0., 8., 4),(0., 2., 21),(0., 1., 11))
    vort_interp = get_callable_scalar_linear(grid_vort, vort_data)
    lavd = lavd_grid_2D(
        fm_n_data, tspan, T, vort_interp, xrav, yrav
    ).astype(np.float32)

    assert np.allclose(lavd,lavd_data)

    lavd_masked = lavd_grid_2D(
        fm_n_data, tspan, T, vort_interp, xrav, yrav, mask=mask_dg, dilate_mask=False
    ).astype(np.float32)

    assert np.allclose(lavd_masked, apply_mask(lavd_data, mask_dg))


def test_ile_2D_func(coords_dg, flow_callable, ile_data, mask_dg):

    vel_func = flow_callable
    x, y = coords_dg
    dx = x[1]
    t0 = 0.

    ile = ile_2D_func(vel_func, x, y, t0=t0, h=dx).astype(np.float32)

    assert np.allclose(ile,ile_data)

    ile_masked = ile_2D_func(
        vel_func, x, y, t0=t0, h=dx, mask=mask_dg, dilate_mask=False
    ).astype(np.float32)

    assert np.allclose(ile_masked, apply_mask(ile_data, mask_dg))

def test_S_eig_2D_func(coords_dg, flow_callable, S_eig_func_data, mask_dg):

    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = flow_callable
    Svals_expected, Svecs_expected = S_eig_func_data
    Svals, Svecs = S_eig_2D_func(vel_func, x, y, t0=t0, h=dx)
    S_expected = reconstruct_matrix(Svals_expected, Svecs_expected)
    S = reconstruct_matrix(Svals.astype(np.float32), Svecs.astype(np.float32))

    assert np.allclose(S, S_expected)

    Svals, Svecs = S_eig_2D_func(
        vel_func, x, y, t0=t0, h=dx, mask=mask_dg, dilate_mask=False
    )
    S_masked = reconstruct_matrix(Svals.astype(np.float32), Svecs.astype(np.float32))

    assert np.allclose(S_masked, apply_mask(S_expected, mask_dg))


def test_S_2D_func(coords_dg, flow_callable, S_data, mask_dg):

    x,y = coords_dg
    dx = x[1]
    t0 = 0.
    vel_func = flow_callable
    S_expected  = S_data
    S = S_2D_func(vel_func, x, y, t0=t0, h=dx).astype(np.float32)

    assert np.allclose(S, S_expected)

    S_masked = S_2D_func(
        vel_func, x, y, t0=t0, h=dx, mask=mask_dg, dilate_mask=False
    ).astype(np.float32)

    assert np.allclose(S_masked, apply_mask(S_expected, mask_dg))

def test_ile_2D_data(coords_dg, vel_data, ile_data, mask_dg):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    u,v = vel_data
    ile = ile_2D_data(u, v, dx, dy).astype(np.float32)

    assert np.allclose(ile, ile_data)

    ile_masked = ile_2D_data(u, v, dx, dy, mask=mask_dg, dilate_mask=False).astype(np.float32)

    assert np.allclose(ile_masked, apply_mask(ile_data, mask_dg))

def test_S_eig_2D_data(coords_dg, vel_data, S_eig_data, mask_dg):

    x,y = coords_dg
    dx = x[1]
    dy = y[1]
    u,v = vel_data
    Svals_expected, Svecs_expected = S_eig_data
    Svals, Svecs = S_eig_2D_data(u,v,dx,dy)
    S_expected = reconstruct_matrix(Svals_expected, Svecs_expected)
    S = reconstruct_matrix(Svals.astype(np.float32), Svecs.astype(np.float32))

    assert np.allclose(S, S_expected)

    Svals_masked, Svecs_masked = S_eig_2D_data(u, v, dx, dy, mask=mask_dg, dilate_mask=False)
    S_masked = reconstruct_matrix(
        Svals_masked.astype(np.float32), Svecs_masked.astype(np.float32))

    assert np.allclose(S_masked, apply_mask(S_expected, mask_dg))

def test_ivd_grid_2D(vort_data, ivd_data):

    vort = vort_data[0, :, :]
    vort_avg = np.mean(vort)
    ivd = ivd_grid_2D(vort, vort_avg).astype(np.float32)

    assert np.allclose(ivd, ivd_data)
