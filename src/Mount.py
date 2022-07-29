import numpy as np
import itertools
import sys
import time

import matplotlib.pyplot as plt

import torch
from torch.fft import fftshift as fftshift_t, fftn as fftn_t, ifftn as ifftn_t

import Grid
import FFT as custom

from Resample import getBufferSize

from scipy.spatial.transform import Rotation
from numpy.fft import fftshift

class Plugin( Grid.Plugin ):
    def __init__( self, size, R, Br, s0 ):
        #self._rho = img_t # fftshifted torch tensor
        self._domainSize = tuple( size for n in range( 3 ) )
        self.initializeDiffractometer( R, Br, s0 )
        return

    def initializeDiffractometer( self, R, Br, s0 ):
        #self._delTheta = 2.*np.pi/3.
        self._delTheta = np.pi/4.
            # largest single rotation, TODO: calculate this better.
        self._axis_dict = { 'X':0, 'Y':1, 'Z':2 }
        self.createBuffer( Br, s0 )
        self.setUpGrid() # inherited from Grid.Plugin
        self.createMask()
        self.getRotationParameters( R, 'XYZ' ) 
        self.prepareToShear( Br )
        return

    def createBuffer( self, Br, s0 ):
        self._diffbuff = getBufferSize( np.diag( Br )/s0, self._domainSize )
        self._padder = torch.nn.ConstantPad3d( 
            list( itertools.chain( 
                *[ [ n, n ] if n > 0 else [ 0, 0 ] for n in self._diffbuff[::-1] ]
            ) ), 0 
        )
        self._skin = tuple( 
            ( 0, N ) if db <= 0 else ( db, N+db ) 
            for db, N in zip( self._diffbuff, self._domainSize ) 
        )
        return

    def createMask( self ):
        mask = np.ones( self._padder( torch.zeros( self._domainSize ) ).shape )
        for num in self._diffbuff:
            if num < 0:
                mask[ -num:num, :, : ] = 0.
            mask = np.transpose( mask, np.roll( [ 0, 1, 2 ], -1 ) )
        self._mask = torch.from_numpy( 1. - mask ).cuda()
        return

    def shearPrincipal( self, ax, shift ):
        self._rho = torch.fft.ifftn( 
            shift * torch.fft.fftn( self._rho, dim=[ ax ], norm='ortho' ), 
            dim=[ ax ], 
            norm='ortho'
        )
        return

    def rotatePrincipal( self, angle, ax=2 ):
        ax1 = (ax+1)%3
        ax2 = (ax+2)%3
        shift1, shift2 = tuple( 
            torch.exp( 1.j * ang * self._gridprod[ax] )
            for ang in [ np.tan(angle/2.), -np.sin(angle) ]
        )
        for x, sh in zip( [ ax1, ax2, ax1 ], [ shift1, shift2, shift1 ] ):
            self.shearPrincipal( x, sh )
        return
    
    def prepareToShear( self, Br ):
        self._Bshear = Br.T / np.diag( Br ).reshape( 1, -1 ).repeat( 3, axis=0 )
        self._shearShift = []
        for n in range( 3 ):
            np1, np2 = (n+1)%3, (n+2)%3
            alpha, beta = self._Bshear[np1,n], self._Bshear[np2,n]
            ax1, ax2 = (n-1)%3, (n+1)%3
            shift_arg = alpha*self._gridprod[ax1] + beta*self._gridprod[ax2]
            shift = torch.exp( 1.j * shift_arg )
            self._shearShift.append( shift )
        return

    def shearResampleObject( self ):
        for ax in range( 3 ):
            self.shearPrincipal( ax, self._shearShift[ax] )
        return
    
    def splitAngle( self, angle ):
        anglist = [ np.sign( angle )*self._delTheta ] * int( np.absolute( angle//self._delTheta ) )
        if angle < 0.:
            anglist[-1] += ( angle%self._delTheta )
        else: 
            anglist.append( angle%self._delTheta )
        return anglist

    def getRotationParameters( self, R, euler_convention='XYZ' ):
        convention = [ self._axis_dict[k] for k in list( euler_convention ) ]
        eulers = Rotation.from_matrix( R ).as_euler( euler_convention )
        self._axes = []
        self._eulers = []
        for ang, ax in zip( eulers[::-1], convention[::-1] ):
            anglist = self.splitAngle( ang )
            self._axes.extend( [ ax ]*len( anglist ) )
            self._eulers.extend( anglist )
        return

    def rotateObject( self ): 
        for ang, ax in zip( self._eulers, self._axes ):
            self.rotatePrincipal( ang, ax )
        return

    def bulkResampleObject( self ):
        self._rhopad = self._padder( fftshift_t( self._rho ) ) # padded object array
        
        n = self._diffbuff[0]
        if n < 0:
            self._rhopad[-n:n,:,:] = custom.ifft1d( 
                ( self._mask * custom.fft1d( self._rhopad, 0 ) )[-n:n,:,:], 
                0 
            )
        elif n > 0:
            self._rhopad[n:-n,:,:] = custom.fft1d( self._rhopad[n:-n,:,:], 0 )
            self._rhopad = custom.ifft1d( self._rhopad, 0 )

        n = self._diffbuff[1]
        if n < 0:
            self._rhopad[:,-n:n,:] = custom.ifft1d( 
                ( self._mask * custom.fft1d( self._rhopad, 1 ) )[:,-n:n,:], 
                1 
            )
        elif n > 0:
            self._rhopad[:,n:-n,:] = custom.fft1d( self._rhopad[:,n:-n,:], 1 )
            self._rhopad = custom.ifft1d( self._rhopad, 1 )

        n = self._diffbuff[2]
        if n < 0:
            self._rhopad[:,:,-n:n] = custom.ifft1d( 
                ( self._mask * custom.fft1d( self._rhopad, 2 ) )[:,:,-n:n], 
                2 
            )
        elif n > 0:
            self._rhopad[:,:,n:-n] = custom.fft1d( self._rhopad[:,:,n:-n], 2 )
            self._rhopad = custom.ifft1d( self._rhopad, 2 )

        self.removeBuffer()
        return

    def removeBuffer( self ):
        self._rho = fftshift_t( 
            self._rhopad[ 
                self._skin[0][0]:self._skin[0][1], 
                self._skin[1][0]:self._skin[1][1], 
                self._skin[2][0]:self._skin[2][1]
            ] 
        )
        return

    def mountObject( self ):
        self.rotateObject()
        self.bulkResampleObject()
        self.shearResampleObject()
        return

    def getMountedObject( self ):
        return self._rho 

    def refreshObject( self, img_t ):
        self._rho = img_t
        return

#=================== end class definition ============================================

if __name__=="__main__":

    #import Demos as demo
    #demo.MountDemo()
    import numpy as np
    from logzero import logger
    from argparse import Namespace
    import h5py as h5
    import matplotlib.pyplot as plt
    from numpy.fft import fftshift
    import torch
    from torch.autograd import Variable
    
    import TilePlot as tp
    from Mount import Plugin as mount
    
    def getSnapshot( title ):
        img_out = fftshift( mnt._rho.cpu() )
        fig = tp.TilePlot(
            tuple( field( np.transpose( img_out, np.roll( [ 0, 1, 2 ], n ) )[:,:,64] ) 
                for field in [ np.absolute, np.angle ] for n in range( 3 ) 
            ),
            ( 2, 3 )
        )[0]
        fig.suptitle( title )
        plt.savefig( '%s.pdf'%title )
        return

    plt.close( 'all' )
    scan = 'scans/dataset_6' 
    with h5.File( '../../BCDISimulations_BETTER.h5', 'r' ) as objdata:
        temp = objdata[ '%s/object'%scan ][:]
        img_shift = fftshift( temp )
        intens = fftshift( objdata[ '%s/intensity'%scan ][:] )
        Br = objdata[ scan ].attrs[ 'Breal' ] # already in detector frame
    s0 = np.absolute( np.linalg.det( Br ) )**( 1./3. )
    R = np.linalg.svd( np.random.rand( 3, 3 ) )[0]
    if np.linalg.det( R ) < 0.:
        R = R[:,::-1]    # right handed rotation

    logger.info( 'Running test for Mounted object. ' )
    mnt = mount( torch.from_numpy( img_shift ).cuda(), R, Br, s0 )
    logger.info( 'Created Mount.Plugin object. ' )
    logger.info( '------------------------------------' )
    
    fig0 = tp.TilePlot(
        tuple( field( np.transpose( temp, np.roll( [ 0, 1, 2 ], n ) )[:,:,64] ) 
            for field in [ np.absolute, np.angle ] for n in range( 3 ) 
        ),
        ( 2, 3 )
    )[0]
    fig0.suptitle( 'Original object' )

#    logger.info( 'Now shearing object arbitrarily along principal directions. ' )
#    for ax in range( 3 ):
#        ax1, ax2 = (ax-1)%3, (ax+1)%3
#        a, b = list( 0.1*( -1. + 2.*np.random.rand( 2 ) ) ) 
#        logger.debug( 'Shear parameters for axis %d = %f, %f'%( ax, a, b ) )
#        shift = torch.exp( 
#            2.j * np.pi * ( a*mnt._gridprod[ax1] + b*mnt._gridprod[ax2] )
#        )
#        ti = time.time()
#        mnt.shearPrincipal( ax, shift )
#        tf = time.time()
#        logger.info( 'Time taken to shear = %f seconds. '%( tf-ti ) )
#    logger.info( 'Done shearing along principal axes. ' )
#    logger.info( '------------------------------------' )
#    getSnapshot( 'Sheared object' )
#
#    logger.info( 'Now rotating about random principal axis. ' )
#    ax = np.random.randint( low=0, high=10000 ) % 3
#    ang_rand = 3.*np.pi/4. * ( -1. + 2.*np.random.rand() ) # small-angle rotations first
#    logger.debug( 'Angle = %f deg, axis = %d'%( ang_rand*180./np.pi, ax ) )
#    ti = time.time()
#    mnt.rotatePrincipal( angle=ang_rand, ax=ax )
#    tf = time.time()
#    logger.info( 'Time taken for rotation = %f seconds. '%( tf-ti ) )
#    logger.info( '------------------------------------' )
#    getSnapshot( 'Rotated object' )
   
    logger.info( 'Now performing user-defined rotation R. ' )
    rv = Rotation.from_matrix( R ).as_rotvec()
    ang = np.linalg.norm( rv )
    ax = rv / ang
    logger.debug( 'Angle = %.3f, axis = %s'%( ang*180./np.pi, str( list( ax ) ) ) )
    logger.debug( 'In the XYZ Euler convention, effectively %d principal rotations'%len( mnt._eulers ) )
    logger.debug( 'Euler angles: %s'%( str( [ ang*180./np.pi for ang in mnt._eulers ] ) ) )
    logger.debug( 'Rotation axes: %s'%( str( mnt._axes ) ) )
    ti = time.time()
    mnt.rotateObject()
    tf= time.time()
    logger.info( 'Time taken for user-defined rotation = %f seconds. '%( tf-ti ) )
    logger.info( '------------------------------------' )
    getSnapshot( 'User-defined-rotation' )

    logger.info( 'Now performing diffractometer bulk resampling. ' )
    ti = time.time()
    mnt.bulkResampleObject()
    tf = time.time()
    logger.info( 'Time taken for bulk resampling = %f seconds. '%( tf-ti ) )
    logger.info( '------------------------------------' )
    getSnapshot( 'Bulk-resampled' )
    
    logger.info( 'Now performing diffractometer shear resampling. ' )
    ti = time.time()
    mnt.shearResampleObject()
    tf = time.time()
    logger.info( 'Time taken for bulk resampling = %f seconds. '%( tf-ti ) )
    logger.info( '------------------------------------' )
    getSnapshot( 'Shear-resampled' )

    logger.info( 'Full mounting sequence with fresh object. ' )
    mnt = mount( torch.from_numpy( img_shift ).cuda(), R, Br, s0 )
    logger.info( 'Created Mount.Plugin object. ' )
    ti = time.time()
    mnt.mountObject()
    tf = time.time()
    logger.info( 'Time taken for full mounting = %f seconds. '%( tf-ti ) )
    logger.info( '------------------------------------' )







