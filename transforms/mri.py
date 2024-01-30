import torch
from typing import List

from typing import List, Callable

from utils import itok, ktoi
from .complex import crc
from .debug import Transform


def simulate_readout_FSE_2d(image : torch.Tensor, ETL : int = 32,
                            dim = 1, probability = 0.25, movefunc : Callable | List[Callable] = None, move_iterable = False):
    """Simulates the readout process according to the FSE acquisition mode.
    ETL: number of lines acquired at a time.
    probability : the probability of the motion occurring.
    movefunc = mode of motion, accepts input from external function.
    move_inplace: True in-place motion, False iterative motion."""

    def __domove(image : torch.Tensor, movefunc : Callable | List[Callable] ) -> torch.Tensor:
        if isinstance(movefunc, List):
            for func in movefunc:
                func = crc(func)
                image = func(image)
        else:
            movefunc = crc(movefunc)
            image = movefunc(image)
        return image
    
    
    def __readout(dst_kspace : torch.Tensor, src_kspace : torch.Tensor, start : int, nline : int) -> torch.Tensor:
        # Always readout at the last dimension.

        if dst_kspace.shape != src_kspace.shape:
            raise ValueError(f'kspace shape not mathching. dst_space.shape = {dst_kspace.shape}, src_kspace.shape = {src_kspace.shape}')

        nd = len(src_kspace.shape)
        end = min(start+nline,src_kspace.shape[-1])
        if nd == 2:
            dst_kspace[:,start:end] = src_kspace[:,start:end]
            return dst_kspace
        if nd == 3:
            dst_kspace[:,:,start:end] = src_kspace[:,:,start:end]
            return dst_kspace
        if nd == 4:
            dst_kspace[:,:,:,start:end] = src_kspace[:,:,:,start:end]
            return dst_kspace

    if dim in [0, -2]:
        image = image.swapdims(-1,-2).contiguous()
    elif dim in [1, -1]:
        pass
    else:
        raise ValueError('dim must in [0,1]')

    kspace = itok(image)
    true_x = image

    # init
    raw_kspace = torch.zeros_like(kspace)
    x = true_x # image domain


    for p in range(0,raw_kspace.shape[-1],ETL):
        # step1 affine
        if torch.rand(1)[0].item() < probability:
            if not move_iterable:
                x = true_x
            x = __domove(x, movefunc=movefunc)
        else:
            pass
        # step2 get kspace
        x_temp_kspace = itok(x)
        # step3 readout
        raw_kspace = __readout(raw_kspace, x_temp_kspace, p, ETL)

    

    raw_image = ktoi(raw_kspace) 

    if dim in [0, -2]:
        raw_image = raw_image.swapdims(-1,-2).contiguous()
    elif dim in [1, -1]:
        pass
    else:
        raise ValueError('dim must in [0,1]')
    
    return raw_image

class FSE_readout(Transform):
    """Simulating the acquisition of FSR sequences in MRI, simulating motion artifact generation."""
    def __init__(self,  ETL : int = 8, dim = 1, probability = 0.25, movefunc : Callable | List[Callable] = None, move_iterable = False,
                 for_keys = None):
        super().__init__(for_keys = for_keys)
        self.ETL = ETL
        self.dim = dim
        self.probability = probability
        self.movefunc = movefunc
        self.move_iterable = move_iterable

    def do(self, x: torch.Tensor):
        x1  = simulate_readout_FSE_2d(x, self.ETL, self.dim, self.probability, self.movefunc, self.move_iterable)
        return x1, x
    

# def _getmp(kspace: np.array):
#     # if self.mps is not None:
#     #     return self.mps
#     SPE, Nc, FE, PE = kspace.shape
#     mps = np.zeros_like(kspace)
#     for slice in range(SPE):
#         mps[slice, :] = EspiritCalib(kspace[slice, :], show_pbar=False).run()
#     return mps


# class Fusioncoil:

#     def __init__(self, keyname: str = 'kspace', _lambda: int = 0):
#         super().__init__()
#         self.keyname = keyname
#         self._lambda = _lambda


#     def __call__(self, x, recon=None):
#         if recon is None:
#             if isinstance(x, List):
#                 y = x.copy()
#                 for i in range(len(y)):
#                     kspace = y[i][self.keyname].numpy()
#                     mps = _getmp(kspace)
#                     onechan = np.sum(kspace * mps, 1)
#                     y[i][self.keyname] = torch.from_numpy(onechan)
#                 return y
#             elif isinstance(x, dict):
#                 y = x.copy()
#                 kspace = y[self.keyname].numpy()
#                 mps = _getmp(kspace)
#                 onechan = np.sum(kspace * mps, 1)
#                 y[self.keyname] = torch.from_numpy(onechan)
#                 return y

#         # if recon == 'sense':
#         #     if isinstance(x, List):
#         #         mps = self.getmps(x[0][self.keyname])
#         #         for i in range(len(x)):
#         #             x[i] = torch.from_numpy(SenseRecon(x[i][self.keyname].numpy(), mps=mps, lamda=self._lambda).run())
#         #
#         #         return x
#         #     elif isinstance(x, dict):
#         #         mps = self.getmps(x[self.keyname])
#         #
#         #         try:
#         #             for i in range(10):
#         #                 x = torch.from_numpy(
#         #                     SenseRecon(x[self.keyname].numpy()[:, i], mps=mps[:, i], lamda=self._lambda).run())
#         #         except Exception as e:
#         #             print(x[self.keyname].shape)
#         #             raise e
#         #         return x

#         typex = type(x)
#         raise TypeError(f"Fusioncoil only receives list[dict[tensor]] or dict[tensor]. But here is a {typex}.")
