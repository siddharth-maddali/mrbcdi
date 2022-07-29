##################################################################
#
#    Routines to create tight-bound object in array. 
#
#    Siddharth Maddali
#    Argonne National Laboratory
#    smaddali@anl.gov
#
#######################################################################

import torch
import numpy as np

class Plugin: 

    def createObjectBB_full( self, n_scan ): # n_scan defined in class MultiReflectionSolver
        """
        Creates phase object within bounding box from FULL set of optimizable variables. 
        """
        self.scl = self.x[-len(self.bragg_condition):][n_scan]
        self.__amp__ = 0.5*( 1. + torch.tanh( self.x[:self.N] / self.activ_a ) )
        self.__u__ = self.x[self.N:(4*self.N)].reshape( 3, -1 )
        phs = self.peaks[n_scan] @ self.__u__
        objBB = self.scl * self.__amp__ * torch.exp( 2.j * np.pi * phs )
        return objBB.reshape( self.cubeSize, self.cubeSize, self.cubeSize )

    def buildFullBB( self ):
        arr1 = [ val*np.ones( self.N ) for val in [ -10.*self.activ_a, 0., 0., 0. ] ]
        arr1.append( np.zeros( len( self.bragg_condition ) ) )
        self.x = torch.tensor( 
            np.concatenate( 
                tuple( arr1 )
            )
        ).cuda() # this is the new array used to build the bound-box object
        self.x[ self.these_only ] = self.x_new # slice optimized variables into original-sized array
        return

    def createObjectBB_part( self, n_scan ):
        """
        Creates phase object within bounding box from RESTRICTED set of optimizable variables. 
        """
        self.buildFullBB()
        objBB = self.createObjectBB_full( n_scan )
        return objBB
