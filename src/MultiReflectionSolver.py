##################################################################
#
#    Multiple  reflection BCDI optimization solver in pure Python. 
#
#    Siddharth Maddali
#    Argonne National Laboratory
#    smaddali@anl.gov
#
#######################################################################

import numpy as np
from tqdm.auto import tqdm
import functools as ftools

import torch
import random

from torch.fft import fftshift as fftshift_t, fftn as fftn_t
from scipy.ndimage import median_filter
from logzero import logger

try:
    from pyfftw.interfaces.numpy_fft import fftshift
except: 
    from numpy.fft import fftshift

from Mount import Plugin as Mount

# The following modules have plugins
import Optimizer, Lattice, Object

import TilePlot as tp # custom plotting

class MultiReflectionSolver( 
        Lattice.Plugin, 
        Object.Plugin,
        Optimizer.Plugin 
    ):

    def __init__( self, 
            database, 
            signal_label='signal', 
            size=128, sigma=3., 
            activation_parameter=0.75,
            learning_rate=0.01, 
            minibatch_plan=None,
            lambda_tv=None,
            medfilter_kernel=3, 
            init_amplitude_state=None,
            scale_method='photon_max', # options are 'photon_max' and 'energy'. Use 'photon_max' for low SNR datasets.
            #ramp_tol=0.1, # tolerance parameter for finite ramp, don't change this
            cuda_device=0
        ):
        self.loadGlobals( database ) #, ramp_tol )
        self.getLatticeBases( *tuple( self.globals[ 'lattice_parms' ] ) )
        self.getRotatedCrystalBases()
        self.determineDimensions( size, sigma )
        self.createObjectBB = self.createObjectBB_full # inherited from Object.Plugin
        self.prepareScans( label=signal_label )
        self.createUnknowns( activation_parameter, init_amplitude_state, scale_method )
        self.initializeOptimizer(
            self.x,
            learning_rate=learning_rate, 
            lambda_tv=lambda_tv, 
            minibatch_plan=minibatch_plan
        )
        self.prepareMedianFilter( medfilter_kernel )
        return

    def prepareMedianFilter( self, kernel ):
        self.mfk = kernel
        return

    def loadGlobals( self, dbase ): #, rt ):
        #self.lattice = lattice_parameters
        self.database = dbase 
        self.globals = dict( self.database[ 'global' ].attrs )
        #self.rt = torch.tensor( rt )
        #self.ramp_const = ( -rt/4. ) * ( 1. + 1./torch.tanh( 2./self.rt ) )
        return

    def determineDimensions( self, size, sigma ):
        self.size = size
        self._domainSize = tuple( self.size for n in range( 3 ) )
        self.cubeSize = np.round( size / sigma ).astype( int )
        self.cubeSize += ( self.cubeSize%2 )
        buff = ( size-self.cubeSize )//2
        self._buffer = torch.nn.ConstantPad3d( tuple( buff for n in range( 6 ) ), 0 )
        return

    def createUnknowns( self, activation_parameter, init_amplitude_state, scale_method ):
        self.activ_a = activation_parameter
        self.N = self.cubeSize**3
        if isinstance( init_amplitude_state, type( None ) ):
            mag = 2.*np.ones( self.N )
        else: 
            mag = init_amplitude_state
        u = np.zeros( 3 * self.N )
        norms = self.setScalingFactors( mag, scale_method )
        self.x = torch.from_numpy( np.concatenate( ( mag, u, norms ) ) ).cuda().requires_grad_()
        return

    def setScalingFactors( self, init, method ):
        norms = []
        init_t = torch.from_numpy( init ).cuda() 
        for n in range( len( self.bragg_condition ) ):
            __amp__ = 0.5*( 1.  + torch.tanh( init_t / self.activ_a ) )
            __u__ = torch.from_numpy( np.zeros( ( 3, self.N ) ) ).cuda()
            phs = self.peaks[n] @ __u__
            objBB = ( __amp__ * torch.exp( 2.j * np.pi * phs ) ).reshape( *( self.cubeSize for n in range( 3 ) ) )
            self.bragg_condition[n][ 'mount' ].refreshObject( fftshift_t( self._buffer( objBB ) ) )
            self.bragg_condition[n][ 'mount' ].mountObject()
            rho_m = self.bragg_condition[n][ 'mount' ].getMountedObject()
            if method=='photon_max':
                frho_m = fftshift_t( fftn_t( fftshift_t( rho_m ), norm='ortho' ) ).detach().cpu().numpy()
                scl = np.sqrt( self.bragg_condition[n][ 'data' ].detach().cpu().numpy().max() / ( np.absolute( frho_m )**2 ).max() )
                norms.append( scl )
            elif method=='energy':
                norms.append(
                    self.bragg_condition[n][ 'data' ].detach().cpu().numpy().sum() / ( np.absolute( rho_m.detach().cpu().numpy() )**2 ).sum()
                )
            else:
                logger.error( 'Should set either \'photon_max\' or \'energy\' for scale_method. ' )
                return []
        return np.sqrt( np.array( norms ) )


    def resetUnknowns( self, x ):
        self.x = torch.from_numpy( x ).cuda().requires_grad_()
        return


    def prepareScans( self, label ):
        self.bragg_condition = []
        self.peaks = []
        self.miller_idx = []
        self.scan_list = [ 'scans/dataset_%d'%m for m in self.database[ 'scans' ].attrs[ 'successful_scans' ] ]
        for scan in self.scan_list:
            self.peaks.append(
                torch.from_numpy( 
                    self.database[ scan ].attrs[ 'peak' ][np.newaxis,:] * 10.
                ).cuda() # convert units from angstrom^-1 to nm^-1, ensures u is determined in nm
            )
            self.miller_idx.append( self.database[ scan ].attrs[ 'miller_idx' ] )
            scan_data = self.database[ '%s/%s'%( scan, label ) ][:]
            R       = self.database[ scan ].attrs[ 'RtoBragg' ]
            Bdet    = self.database[ scan ].attrs[ 'Bdet' ]
            Br      = self.database[ scan ].attrs[ 'Breal' ] # columns are in units of nm
            mnt     = Mount( self.size, Bdet.T @ R, Br, self.globals[ 'step_cubic' ] ) 
                # this initializes the diffractometer
            bcond = { 
                'data':torch.from_numpy( fftshift( np.sqrt( scan_data ) ) ).cuda(), 
                'mount':mnt
                }
            self.bragg_condition.append( bcond )
        self.d_spacing  = self.getPlaneSeparations()
        self.d_jumps    = [ mg**2 * pk.T for mg, pk in zip( self.d_spacing, self.peaks ) ]
        return

    def getObjectInMount( self, n ):
        obj = self.createObjectBB( n )
        self.bragg_condition[n][ 'mount' ].refreshObject( fftshift_t( self._buffer( obj ) ) )
        self.bragg_condition[n][ 'mount' ].mountObject()
        rho_m = self.bragg_condition[n][ 'mount' ].getMountedObject()
        return rho_m

    def centerObject( self ):
        state = self.x[:self.N].reshape( *( self.cubeSize for n in range( 3 ) ) )
        amp = 0.5*( 1. + torch.tanh( state / self.activ_a ) ).detach().cpu().numpy()
        u = self.x[self.N:(4*self.N)].reshape( 3, -1 ).detach().cpu().numpy()
        ux, uy, uz = tuple( arr.reshape( *( self.cubeSize for n in range( 3 ) ) ) for arr in u )
        grid = np.mgrid[ 
            -self.cubeSize//2:self.cubeSize//2, 
            -self.cubeSize//2:self.cubeSize//2, 
            -self.cubeSize//2:self.cubeSize//2
        ]
        shift = [ -np.round( ( arr*amp ).sum() / amp.sum() ).astype( int ) for arr in grid ]
        state_c = np.roll( state.detach().cpu().numpy(), shift, axis=[ 0, 1, 2 ] )
        ux_c, uy_c, uz_c = tuple( np.roll( arr, shift, axis=[ 0, 1, 2 ]  ) for arr in [ ux, uy, uz ] )
        xi = self.x[-len( self.bragg_condition ):].detach().cpu().numpy() 
        my_x = np.concatenate( 
            tuple( 
                arr.ravel()
                for arr in [ state_c, ux_c, uy_c, uz_c, xi ]
            )
        )
        new_x = torch.from_numpy( my_x ).requires_grad_().cuda()
        with torch.no_grad():
            self.x.copy_( new_x )
        return

    # restricted optimization methods begin from here. 

    def setUpRestrictedOptimization( self, new_plan, amp_threshold=0.1, lambda_tv=None, learning_rate=None, median_filter=False ):
        self.getSupportVars( amp_threshold, median_filter=median_filter )
        self.createObjectBB = self.createObjectBB_part

        # retain old values of optimization parameters if new ones not specified
        if not isinstance( lambda_tv, type( None ) ):
            self._ltv_ = lambda_tv
        if not isinstance( learning_rate, type( None ) ):
            self.lr = learning_rate

        self.initializeOptimizer(
            self.x_new, # now optimizing over only support voxels. 
            learning_rate=self.lr, 
            lambda_tv=self._ltv_, 
            minibatch_plan=new_plan # should in general be different from old plan
        )
        return

    def getSupportVars( self, amp_threshold, median_filter ):
        self.bin = [] # stores calculated values
        if median_filter:
            my_x = self.medianFilter()
        else: 
            my_x = self.x.detach().cpu().numpy()
        ln = my_x.size
        self.bin.append( my_x )
        state = my_x[:self.N]
        amp = 0.5*( 1.  + np.tanh( state / self.activ_a ) )
        here_c = np.where( amp > amp_threshold )[0] # these are the support voxels
        here_c = np.concatenate( tuple( n*self.N + here_c for n in range( 4 ) ) ) # extend for all u's
        here_c = np.concatenate( ( here_c, np.array( [ ln-len( self.bragg_condition )+n for n in range( len( self.bragg_condition ) ) ] ) ) )
        self.bin.append( here_c )
        my_new_x = my_x[ here_c ] # only these voxels are optimized from here on. 
        self.x_new = torch.from_numpy( my_new_x ).cuda().requires_grad_()
        these = np.zeros( my_x.size )
        these[ here_c ] = 1.
        self.these_only = torch.tensor( these, dtype=torch.bool ).cuda() # indexes of optimized variables
        return

    def medianFilter( self ):
        my_x = self.x.detach().cpu().numpy()
        arr = my_x[:-len( self.bragg_condition ) ]
        scalers = my_x[ -len( self.bragg_condition ): ]
        arr_by4 = arr.reshape( 4, -1 )
        state, ux, uy, uz = tuple( ar.reshape( *( self.cubeSize for n in range( 3 ) ) ) for ar in arr_by4 )
        amp = 0.5*( 1.  + np.tanh( state / self.activ_a ) )
        ux, uy, uz = tuple( median_filter( amp*ar, size=self.mfk ) for ar in [ ux, uy, uz ] )
        arr_out = np.concatenate( tuple( ar.ravel() for ar in [ state, ux, uy, uz, scalers ] ) )
        with torch.no_grad():
            self.x.copy_( torch.from_numpy( arr_out ).cuda() )
        return arr_out

        

#============================= end class definition ===============================

if __name__=="__main__":

    import numpy as np
    import h5py as h5
    
    from logzero import logger
    import matplotlib.pyplot as plt
    import TilePlot as tp
    from torch.fft import fftshift as fftshift_t, fftn as fftn_t


    datafile = '../../BCDISimulations_general_FCC_Au.h5'
    minibatch_plan={ 
        #'minibatches':[ 20, 20, 20, 20, 20, 20, 20 ], 
        #'minibatches':[ 100, 80, 60, 40, 20, 10, 5 ], 
        'minibatches':[ 100, 80, 60, 40, 20, 20, 10 ], 
        'minibatch_size':[ 5, 6, 6, 6, 6, 6, 6, 7 ], 
        'iterations_per_minibatch':[ 3, 6, 12, 25, 50, 100, 500 ]
    }

    logger.info( 'Starting test for multi-reflection solver. ' )
    logger.info( '------------------------------------' )
    with h5.File( datafile, 'r' ) as data:
        solver = MultiReflectionSolver( 
            database=data,
            signal_label='signal/photon_max_100000.0',
            sigma=3.55, # CHEATING HERE FOR DEMO'S SAKE: this window is just snug enough for this object
            activation_parameter=2.,  
            minibatch_plan=minibatch_plan,
            lambda_tv=1.e-6
        )

    

    logger.info( 'Global experimental parameters: ' )
    for key, value in solver.globals.items(): 
        logger.info( '\t\t%s: %s'%( key, str( value ) ) )

    logger.info( 'Optimization plan: ' )
    logger.info( '\t\tMini-batches: ' + str( minibatch_plan[ 'minibatches' ] ) )
    logger.info( '\t\tMini-batch size: ' + str( minibatch_plan[ 'minibatch_size' ] ) )
    logger.info( '\t\tIteration per minibatch: ' + str( minibatch_plan[ 'iterations_per_minibatch' ] ) )

    logger.info( 'Running optimizer...' )
    scalers = solver.run()
    solver.run()

    with h5.File( 'Multireflection_result.h5', 'w' ) as fid:
        fid.create_dataset( 'result', data=solver.x.detach().cpu() )
        fid.create_dataset( 'error', data=np.array( solver.error ) )

    plt.close( 'all' )
    plt.ioff()
    logger.info( 'Plotting cost function. ' )
    plt.figure()
    plt.semilogy( solver.error )
    plt.grid()
    plt.xlabel( 'Iteration' )
    plt.ylabel( r'$\mathcal{L}(A, \bf{u})$' )
    plt.savefig( 'result_costFunction.pdf' )

    logger.info( 'Plotting reconstructed electron density. ' )
    fig, im, ax = tp.TilePlot( 
        tuple(
            np.transpose( 
                solver.__amp__.detach().cpu().reshape( *( solver.cubeSize for n in range( 3 ) ) ), 
                np.roll( [ 0, 1, 2 ], n ) 
            )[:,:,solver.cubeSize//2]
            for n in range( 3 )
        ), 
        ( 1, 3 )
    )
    for n, ttl in enumerate( [ 'XY', 'YZ', 'ZX' ] ):
        ax[n].set_title( ttl )
    fig.savefig( 'result_Amplitude.pdf' )

    logger.info( 'Plotting lattice distortions. ' )
    #u = solver.x[solver.N:-len( solver.bragg_condition)].reshape( 3, -1 ).detach().cpu() 
    u = solver.__u__.reshape( 3, -1 ).detach().cpu() 
    fig, im, ax = tp.TilePlot( 
        tuple( 
            arr.reshape( *( solver.cubeSize for p in range( 3 ) ) )[:,:,solver.cubeSize//2] 
            for arr in u 
        ), 
        ( 1, 3 ), 
        color_scales=True 
    )
    for n, ttl in enumerate( [ r'$u_x$', r'$u_y$', r'$u_z$' ] ):
        ax[n].set_title( ttl )
    fig.savefig( 'result_U.pdf' )

    logger.info( 'Plotting inferred diffraction patterns. ' )
    for n, scan in enumerate( solver.scan_list ):
        data_n = fftshift_t( solver.bragg_condition[n][ 'data' ] ).detach().cpu()
        data_n[ np.where( data_n <=0. ) ] = data_n[ np.where( data_n > 0. ) ].min()
        rho_n = solver.getObjectInMount( n )
        frho_n = fftshift_t( fftn_t( rho_n, norm='ortho' ) ).detach().cpu()
        intens_n = np.absolute( frho_n )
        fig, im, ax = tp.TilePlot( 
            tuple( arr[:,:,arr.shape[-1]//2] for arr in [ data_n, intens_n ] ), 
            ( 1, 2 ), 
            log_norm=True, 
            color_scales=True
        )
        ax[0].set_ylabel( scan )
        fig.savefig( ('result_%s.pdf'%scan).replace( 'scans/', '' ) )
    

    plt.ion()
