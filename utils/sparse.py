# coo matrix


from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, sparse_coo_tensor
from torch.special import i0
from torchkbnufft._nufft.utils import validate_args


def build_torch_spmatrix(
    omega: torch.Tensor,
    numpoints: Sequence[int],
    im_size: Sequence[int],
    grid_size: Sequence[int],
    n_shift: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> Tensor:
    """Builds a sparse matrix with the interpolation coefficients.

    Args:
        omega: An array of coordinates to interpolate to (radians/voxel).
        numpoints: Number of points to use for interpolation in each dimension.
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        n_shift: Number of points to shift for fftshifts.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        A scipy sparse interpolation matrix.
    """
    spmat = -1

    ndims = omega.shape[0]
    klength = omega.shape[1]

    # calculate interpolation coefficients using kb kernel
    def interp_coeff_torch(om, npts, grdsz, alpha, order):
        gam = 2 * torch.pi / grdsz
        interp_dist = om / gam - torch.floor(om / gam - npts / 2)
        Jvec = Jvec = torch.arange(1, npts + 1, device=om.device).view(1, npts)
        kern_in = -1 * Jvec + interp_dist.unsqueeze(1)

        cur_coeff = torch.zeros_like(kern_in, dtype=torch.complex128)
        indices = torch.abs(kern_in) < npts / 2
        bess_arg = torch.sqrt(1 - (kern_in[indices] / (npts / 2)) ** 2)
        denom = i0(alpha)
        cur_coeff[indices] = i0(alpha * bess_arg) / denom
        cur_coeff = cur_coeff.real

        return cur_coeff, kern_in

    full_coef = []
    kd = []
    for (
        it_om,
        it_im_size,
        it_grid_size,
        it_numpoints,
        it_om,
        it_alpha,
        it_order,
    ) in zip(omega, im_size, grid_size, numpoints, omega, alpha, order):
        # get the interpolation coefficients
        coef, kern_in = interp_coeff_torch(
            it_om, it_numpoints, it_grid_size, it_alpha, it_order
        )

        gam = 2 * torch.pi / it_grid_size
        phase_scale = 1j * gam * (it_im_size - 1) / 2

        phase = torch.exp(phase_scale * kern_in)
        full_coef.append(phase * coef)

        # nufft_offset
        koff = torch.floor(it_om / gam - it_numpoints / 2).unsqueeze(1)
        Jvec = torch.arange(1, it_numpoints + 1, device=omega.device).view(1, it_numpoints)
        kd.append((Jvec + koff) % it_grid_size + 1)

    for i in range(len(kd)):
        kd[i] = (kd[i] - 1) * torch.prod(torch.tensor(grid_size[i + 1 :], device=omega.device))

    # build the sparse matrix
    kk = kd[0]
    spmat_coef = full_coef[0]
    for i in range(1, ndims):
        Jprod = int(torch.prod(torch.tensor(numpoints[: i + 1])))
        # block outer sum
        kk = (kk.unsqueeze(1) + kd[i].unsqueeze(2)).view(klength, Jprod)
        # block outer prod
        spmat_coef = (spmat_coef.unsqueeze(1) * full_coef[i].unsqueeze(2)).view(klength, Jprod)

    # build in fftshift
    phase = torch.exp(1j * omega.T @ n_shift.unsqueeze(1))
    spmat_coef = spmat_coef.conj() * phase

    # get coordinates in sparse matrix
    trajind = torch.arange(klength, device=omega.device).unsqueeze(1).repeat(1, int(torch.prod(torch.tensor(numpoints))))

    # build the sparse matrix
    spmat = sparse_coo_tensor(indices= (trajind.flatten(), kk.flatten()),
                              values=spmat_coef.flatten(),
                              size=(klength, torch.prod(grid_size)),
                              device = omega.device
    )

    return spmat


def calc_tensor_spmatrix_torch(
    omega: Tensor,
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""Builds a sparse matrix for interpolation.

    This builds the interpolation matrices directly from scipy Kaiser-Bessel
    functions, so using them for a NUFFT should be a little more accurate than
    table interpolation.

    This function has optional parameters for initializing a NUFFT object. See
    :py:class:`~torchkbnufft.KbNufft` for details.

    * :attr:`omega` should be of size ``(len(im_size), klength)``,
      where ``klength`` is the length of the k-space trajectory.

    Args:
        omega: k-space trajectory (in radians/voxel).
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for fftshift. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on. Default:
            ``torch.device('cpu')``

    Returns:
        2-Tuple of (real, imaginary) tensors for NUFFT interpolation.

    Examples:

        >>> data = torch.randn(1, 1, 12) + 1j * torch.randn(1, 1, 12)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> spmats = tkbn.calc_tensor_spmatrix(omega, (8, 8))
        >>> adjkb_ob = tkbn.KbNufftAdjoint(im_size=(8, 8))
        >>> image = adjkb_ob(data, omega, spmats)
    """
    if not omega.ndim == 2:
        raise ValueError("Sparse matrix calculation not implemented for batched omega.")
    (
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        order,
        alpha,
        dtype,
        device,
    ) = validate_args(
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        kbwidth,
        order,
        omega.dtype,
        omega.device,
    )
    coo = build_torch_spmatrix(
        omega=omega,
        numpoints=numpoints,
        im_size=im_size,
        grid_size=grid_size,
        n_shift=n_shift,
        order=order,
        alpha=alpha,
    )

    values = coo.data
    indices = torch.stack((coo.row_indices, coo.col_indices), dim=0)

    inds = torch.tensor(indices, dtype=torch.long, device=device)
    real_vals = torch.tensor(values.real, dtype=dtype, device=device)
    imag_vals = torch.tensor(values.imag, dtype=dtype, device=device)
    shape = coo.shape

    interp_mats = (
        torch.sparse.FloatTensor(inds, real_vals, torch.Size(shape)),  # type: ignore
        torch.sparse.FloatTensor(inds, imag_vals, torch.Size(shape)),  # type: ignore
    )

    return interp_mats