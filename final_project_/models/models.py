from kernels import (Linear,
                     Polynomial,
                     MaternOneHalf,
                     MaternFiveHalves,
                     MaternThreeHalves,
                     Gaussian,
                     Wendland
                     )

from typing import Callable
from flax import linen as nn
from jax import numpy as jnp
import jax
from models.mutils import get_grid
from numpy.polynomial import (chebyshev,
                              laguerre,
                              legendre,
                              hermite)


#### pass t 

# main model 
class NOpModelNd(nn.Module):
    integration_kernels: tuple[object] # i.e  [Linear, Polynomial, MaternOneHalf, Gaussian]
    interpolation_kernel: object = Gaussian
    non_linearity: Callable = nn.gelu
    dense_init: Callable = nn.initializers.he_normal()
    quadrature_fn: Callable = chebyshev.chebgauss
    num_quad_pts: int = 10
    width: int = 128
    kernel_init: Callable = nn.initializers.constant(1) 
    residual_block: bool = True
    channel_lift: bool = True
    channel_lift_size: int = 128
    num_output_layers: int = 1
    conv: bool = True # versus dense layers in integration block
    secs_per_fn: int = 1 ### to make the grid
    
    @nn.compact
    def __call__(self,f_x, s):
        ndims = len(f_x.shape[1:])
        resolutions = f_x.shape[1:]
        x = get_grid(resolutions, secs_per_fn=self.secs_per_fn) ### for benchmarks, this is a uniform grid from 0,1 w/ given resolutions
        t,w = self.quadrature_fn(self.num_quad_pts)
        t = jnp.array(jnp.meshgrid(*[t]*ndims)) 
        t = t.reshape(len(t),-1).T  ### i.e. (100, 2) for 2d and 10 quad pts
        w = w.repeat(t.shape[0]/self.num_quad_pts)[:,None] 
        f_t = KernelIntNd()(f_x, x, t) ### takes from (batch, nx, ny, ...) to (batch, nt*nt...)
        ## concatenate location values
        f_t = jnp.concatenate((f_t[...,None], 
                             t[None].repeat(f_t.shape[0],axis=0)),
                            axis=-1)

        # cheaper to channel lift after kernel integration     
        if self.channel_lift:
            h_t = nn.Dense(self.channel_lift_size, name='channel lifting')(f_t)
        else: 
            h_t = f_t

        # stacked integration blocks 
        for kernel in self.integration_kernels:
            h_t_in = h_t
            h_t = IntBlockNd(kernel=kernel,
                           non_linearity=self.non_linearity,
                           kernel_init=self.kernel_init,
                          )(h_t_in, t, w)
            
            # residual block
            if self.residual_block: 
                if self.conv:
                    ### 1d conv (channels last convention in flax)
                    h_t_in = nn.Conv(self.channel_lift_size, kernel_size=(1,))\
                        (h_t_in) # (batch, grid, channel_lift_size) --> same

                else: 
                    h_t_in = nn.Dense(self.channel_lift_size)(h_t_in)

                h_t = self.non_linearity(h_t + h_t_in)

            else:
                h_t = self.non_linearity(h_t)

        
        h_t = nn.Dense(1)(h_t) # (batch, nodes, k) ----> (batch, nodes, 1)
        
        # small neural network to predict h_s from h_t
        out_shape = s.shape[:ndims]
        s = s.reshape(-1,)[None].repeat(h_t.shape[0], axis=0)
        h_t = jnp.concatenate((h_t[...,0], s), axis=-1)
        h_s = h_t
        for i in range(self.num_output_layers):
            h_s = nn.Dense(self.width)(h_s)
            h_s = self.non_linearity(nn.Dense(self.width)(h_s))

        h_s = nn.Dense(int(s.shape[1]/ndims))(h_s)
        return  h_s.reshape(len(h_s), *out_shape)
    

class KernelIntNd(nn.Module):  
    kernel_init: Callable = nn.initializers.constant(1) 
    kernel: Callable = Gaussian
    @nn.compact
    def __call__(self, h_x, x, t): 
        '''
        for 1d
        h_x --> (batch,nx), x --> (nx,1), 
                            t --> (nodes,1)
        for 2d
        h_x --> (batch,nx,ny), x --> (nx,ny,2), 
                               t --> (nodes,nodes, 2)
        for 3d
        h_x --> (batch,nx,ny,nz), x --> (nx,ny,nz,3), 
                                  t --> (nodes,nodes,nodes, 3)
        '''
        
        ## t is what we're projecting to.. 
        ndims = len(h_x.shape[1:])
        resolutions = h_x.shape[1:]
        batch_size= h_x.shape[0]
        interp_kparams = self.param(f'kparams_1', 
                                    self.kernel_init,
                                    (self.kernel.num_trainable_params,)) 
        k = self.kernel(*interp_kparams)
        x = x.reshape(-1,ndims)
        t = t.reshape(-1,ndims)
        
        K_xx = k.matrix(x,x) 
        K_xx_var = self.param(f'k_var', 
                           self.kernel_init, 
                           (1,))
        # K_xx_var = 1e-6
        K_tx= k.matrix(t,x)
        I = jnp.identity(x.shape[0])
        # K_xx_inv = jnp.linalg.inv(K_xx + K_xx_var * I)
        K_xx_inv = jax.scipy.linalg.inv(K_xx + K_xx_var * I)
        h_t = h_x.reshape(batch_size,-1) @ (K_tx @ K_xx_inv).T
        return h_t
    

class IntBlockNd(nn.Module):
    kernel: object = Gaussian
    non_linearity: Callable = nn.gelu
    kernel_init: Callable = nn.initializers.constant(1)
    
    @nn.compact
    def __call__(self, h_t, t, w): 
        nu_kparams= self.param(f'nu_kparams', 
                               self.kernel_init, 
                               (self.kernel.num_trainable_params,))
        nu = self.kernel(*nu_kparams)
        h_t = h_t.transpose(0,2,1) @ (nu.matrix(t,t) * w.T)
        return h_t.transpose(0,2,1)