import numpy as np
import torch

#===========================================================================================================

def getBufferSize( coef_of_exp, shp ):
    bufsize = np.array( 
        [
            np.round( 0.5* n * ( 1./c - 1. ) ).astype( int )
            for n, c in zip( 
                shp, coef_of_exp
            )
        ]
    )
    return bufsize

#===========================================================================================================

