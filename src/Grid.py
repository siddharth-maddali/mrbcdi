import numpy as np
import torch
from numpy.fft import fftshift

class Plugin: 

#    def __init__( self, dom ):
#        """
#        Constructor for debug purposes only, should be commented out 
#        when inheriting. 
#        """
#        self._domainSize = dom
#        self.setUpGrid()
#        return

    def setUpGrid( self ):
        grid_np = np.mgrid[ 
            (-self._domainSize[0]//2.):(self._domainSize[0]//2.), 
            (-self._domainSize[1]//2.):(self._domainSize[1]//2.), 
            (-self._domainSize[2]//2.):(self._domainSize[2]//2.) 
        ]
        self._gridprod = [ 
            torch.from_numpy( 
                fftshift( 
                    2. * np.pi * grid_np[(n+2)%3] * grid_np[(n+1)%3] / self._domainSize[n] 
                )
            ).cuda()
            for n in range( 3 ) 
        ]
        return

#=================== end class definition ============================================

if __name__=="__main__": # run this interactively in ipython

    #import Demos as demo
    #demo.GridDemo()
    from logzero import logger 

    logger.info( 'Running test for Grid object. ' )
    from Grid import Plugin as grid

    sz = ( 128, 128, 128 )
    logger.info( 'Domain size = ( %d, %d, %d )'%sz )
    try: 
        grd = grid( sz )
        logger.info( 'Created Grid.Plugin object. ' )
    except: 
        logger.error( 'No constructor defined for Grid.Plugin.' )
