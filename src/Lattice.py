##########################################################################
#
#    Lattice.py: 
#        Plugin for calculations involving 3D Bravais lattices. 
#
#    Siddharth Maddali
#    Dec 2021 
#    Argonne National Laboratory
#    smaddali@anl.gov
#
##########################################################################

import numpy as np
import torch

class Plugin: 
#    def __init__( self, a, b, c, al, bt, gm ):
#        """
#        a, b, c: lengths of lattice basis vectors (nm)
#        α, β, γ: Angular separations of basis vectors (deg)
#        For example, a cubic lattice has: a = b = c and α = β = γ = 90°
#        """
#        self.getLatticeBases( a, b, c, al, bt, gm )
#        return

    def getLatticeBases( self, a, b, c, al, bt, gm ):
        a, b, c = tuple( lat/10. for lat in [ a, b, c ] )                  # converting from Angstrom to nanometers
        al, bt, gm = tuple( np.pi/180. * ang for ang in [ al, bt, gm ] )    # converting from degrees to radians
        p = ( np.cos( al ) - np.cos( bt )*np.cos( gm ) ) / (  np.sin( bt )*np.sin( gm ) )
        q = np.sqrt( 1. - p**2 )
        self.basis_real = np.array( 
            [ 
                [ a , b*np.cos( gm ), c*np.cos( bt ) ], 
                [ 0., b*np.sin( gm ), c*p*np.sin( bt ) ], 
                [ 0., 0., c*q*np.sin( bt ) ]
            ]
        )
        if np.linalg.det( self.basis_real ) < 0.: # convert to right-handed
            self.basis_real[-1,-1] *= -1.

        self.basis_real = torch.from_numpy( self.basis_real ).cuda() # used for unit cell modulo
        self.basis_reciprocal = torch.linalg.inv( self.basis_real.T )
        self.metricTensor = self.basis_real.T @ self.basis_real
        self.planeStepTensor = torch.linalg.inv( self.metricTensor )
        self.d_jumps_unitcell =  1. / torch.sqrt( self.planeStepTensor.sum( axis=0 ) )
        return


    def getPlaneSeparations( self ):
        idx_arr = torch.from_numpy( np.array( [ list( idx ) for idx in self.miller_idx ] ).T ).cuda()   # self.miller_idx inherited from MultireflectionSolver
        d = 1. / torch.sqrt( ( idx_arr * ( self.planeStepTensor @ idx_arr ) ).sum( axis=0 ) )  
        return d                                                                            

    def getRotatedCrystalBases( self ):
        R = torch.from_numpy( self.database[ 'global' ].attrs[ 'Rcryst' ] ).cuda()
        self.basis_real = R @ self.basis_real
        self.basis_reciprocal = R @ self.basis_reciprocal
        return




########################### end class definition #########################

########################### Test script ##################################

if __name__=="__main__":
   
    import numpy as np
    from logzero import logger
    import sys

    peaks = np.array( 
        [ 
            [ 1., 0., 0. ], 
            [ 1., 1., 0. ], 
            [ 1., 1., 1. ], 
            [ 2., 0., 0. ], 
            [ 2., 1., 0. ], 
            [ 2., 1., 1. ], 
            [ 2., 2., 0. ], 
            [ 2., 2., 1. ]
        ]
    ).T
    logger.info( 'Plane Miller indices: %d'%( peaks.shape[-1] ) )
   
    a, b, c, al, bt, gm = 3.97, 3.97, 3.97, 90., 90., 90.       # cubic crystal
#    a, b, c, al, bt, gm = 4.08, 3.97, 2.98, 53., 49., 62.       # triclinic crystal
#    a, b, c, al, bt, gm = 3.98, 3.98, 2.53, 90., 90., 120.      # hexagonal crystal

    try: 
        lat = Plugin( a, b, c, al, bt, gm )
    except: 
        logger.error( 'Constructor not found for class Plugin. ' )
        sys.exit()

    lat.miller_idx = peaks.T
    d = lat.getPlaneSeparations()

    logger.info( str( peaks.T ) )
    logger.info( 'd-spacings (nm): ' + str( d ) )


