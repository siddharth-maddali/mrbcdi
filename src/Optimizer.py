#######################################################################################
#
#    Optimizer.py: 
#        Contains a plugin for optimizer methods for the multi-reflection BCDI 
#        problem
#
#    Siddharth Maddali
#    Argonne National Laboratory
#    Nov 2021
#    smaddali@anl.gov
#
#########################################################################################

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import torch
from torch.fft import fftshift as fftshift_t, fftn as fftn_t
import random

# debug imports
import numpy as np
import TilePlot as tp

def TV3D( arr ):
    """
    Calculates total variation regularization of input3D array, 
    as the sum of total variation along each axis.
    """
    return torch.mean( 
        torch.ravel( torch.abs( arr[1:,:,:] - arr[:-1,:,:] ) ) + 
        torch.ravel( torch.abs( arr[:,1:,:] - arr[:,:-1,:] ) ) + 
        torch.ravel( torch.abs( arr[:,:,1:] - arr[:,:,:-1] ) )
    )


class Plugin: 

    def initializeOptimizer( self, 
        optim_var, # optimize over these variables
        learning_rate, lambda_tv, 
        minibatch_plan, default_iterations=3000 
    ):
        """
        minibatch_plan format: 

        {
            'minibatches':[ list of <N> ]
            'minibatch_size':[ list of <N> ]
            'iterations_per_minibatch':[ list of <N> ]
        }

        The lists range over the number of epochs. A singleton list means only one epoch. 
        Minibatches are automatically and randomly generated.
        """
        self.lr = learning_rate
        try:
            dummy = len( self.error ) # do nothing new if this array already exists
        except: 
            self.error = []
        self.optimizer = torch.optim.Adam( [ optim_var ], lr=self.lr ) 
        if isinstance( lambda_tv, type( None ) ):
            self._lfun_ = self.lossfn
        else: 
            self._ltv_ = lambda_tv  # warning: no serious type checking happening here...
            self._lfun_ = self.lossTV
        if isinstance( minibatch_plan, type( None ) ):  # mini-batch optimization not requested
            minibatch_plan = { 
                'minibatches':[ 1 ],                                #  1 epoch
                'minibatch_size':[ len( self.bragg_condition ) ],   # batch optimization
                'iterations_per_minibatch':[ default_iterations ]
            }
        self.buildCustomPlan( minibatch_plan )
        return

    def buildCustomPlan( self, minibatch_plan ):
        N = list( range( len( self.peaks ) ) )
        scheme = [ 
            [ [ sz, it ] ] * num
            for sz, it, num in zip( 
                minibatch_plan[ 'minibatch_size' ], 
                minibatch_plan[ 'iterations_per_minibatch' ], 
                minibatch_plan[ 'minibatches' ]
            )
        ]
        self.optimization_plan = [ 
            [
                [ random.sample( N, sch[0] ), sch[1] ]
                for sch in epoch
            ]
            for epoch in scheme
        ]
        return

    def unitCellClamp( self, tol=1.e-6 ): 
        """
        Clamps distortions to crystallographic unit cell.
        """
        self.__u__ = self.x[self.N:-len(self.bragg_condition)].reshape( 3, -1 )
        temp = torch.linalg.solve( self.basis_real, self.__u__ )
        with torch.no_grad():
            temp[:] = temp.clamp( -0.5+tol, 0.5-tol ) # keep it just within the unit cell; 0.5 ==> on the border of the next cell!
            self.x[self.N:-len(self.bragg_condition)].copy_( ( self.basis_real @ temp ).ravel() )
        return


    def lossfn( self ):
        return sum( self.losses )

    def penaltyTV( self ):
        mag = self.x[:self.N].reshape( *( self.cubeSize for p in range( 3 ) ) ) # regularizing the state, not the e-density.
        tvmag = TV3D( mag )
        return tvmag

    def lossTV( self ):
        return self.lossfn() + self._ltv_*self.penaltyTV()

    def run( self, epochs=3000 ):
        for epoch in tqdm( 
            self.optimization_plan, 
            desc='Epoch          ', 
            total=len( self.optimization_plan ) 
        ):
            for batch in tqdm( epoch, desc='Batch/minibatch', total=len( epoch ), leave=False ):
                for iteration in tqdm( range( batch[1] ), desc='Iteration      ', leave=False ):
                    self.optimizer.zero_grad()
                    self.losses = []
                    for n in batch[0]:
                        rho_m = self.getObjectInMount( n )
                        frho_m = fftn_t( rho_m, norm='ortho' )
                        self.losses.append( 
                            torch.mean( ( torch.abs( frho_m ) - self.bragg_condition[n][ 'data' ] )**2 )
                        )
                    self.loss_total = self._lfun_()
                    self.loss_total.backward()
                    self.optimizer.step()
                    self.error.append( float( self.loss_total.cpu() ) )
                    self.unitCellClamp()
            self.medianFilter()

        return
