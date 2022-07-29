import torch
from torch.fft import fftshift as fftshift_t, fftn as fftn_t, ifftn as ifftn_t


def fft1d( arr, n ):
    return fftshift_t( 
        fftn_t( 
            fftshift_t( 
                arr, dim=[n] 
            ), 
            dim=[n], 
            norm='ortho' 
        ), 
        dim=[n] 
    )

def ifft1d( arr, n ):
    return fftshift_t( 
        ifftn_t( 
            fftshift_t( 
                arr, dim=[n] 
            ), 
            dim=[n], 
            norm='ortho' 
        ), 
        dim=[n] 
    )
