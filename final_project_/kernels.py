from jax import numpy as jnp
from tensorflow_probability.substrates.jax.math import psd_kernels 

class Wendland: 
    def __init__(self,alpha):
        self.alpha = alpha
        self.d = 1
    def apply(self, x1,x2):
        r = jnp.sqrt((x1 - x2)**2)
        term1 = (1 - r/self.alpha)**(self.d+2)
        term2 = (1 + (self.d/self.alpha) * r + (self.d/(3*self.alpha**2)) * r**2)
        return jnp.where(r <= self.alpha, term1 * term2, 0.0)
    def matrix(self, v1, v2):
        X1,X2 = jnp.meshgrid(v1[:,0], v2[:,0])
        return self.apply(X1,X2)

    num_trainable_params = 1

class MaternOneHalf(psd_kernels.MaternOneHalf):
    def __init__(self, amplitude, length_scale):
        super().__init__(amplitude=amplitude, length_scale=length_scale)   
    num_trainable_params = 2
    
class MaternThreeHalves(psd_kernels.MaternThreeHalves):
    def __init__(self, amplitude, length_scale):
        super().__init__(amplitude=amplitude, length_scale=length_scale)  
    num_trainable_params = 2
    
class MaternFiveHalves(psd_kernels.MaternFiveHalves):
    def __init__(self, amplitude, length_scale):
        super().__init__(amplitude=amplitude, length_scale=length_scale)   
    num_trainable_params = 2
    
class Gaussian(psd_kernels.ExponentiatedQuadratic):
    def __init__(self, amplitude, length_scale):
        super().__init__(amplitude=amplitude, length_scale=length_scale)  
    num_trainable_params = 2
    
class Linear(psd_kernels.Linear):
    def __init__(self, bias_amplitude, slope_amplitude, shift):
        super().__init__(bias_amplitude=bias_amplitude, slope_amplitude=slope_amplitude, shift=shift)  
    num_trainable_params = 3
    
class Polynomial(psd_kernels.Polynomial):
    def __init__(self, bias_amplitude, slope_amplitude, shift, exponent):
        super().__init__(bias_amplitude=bias_amplitude, slope_amplitude=slope_amplitude, shift=shift, exponent=exponent)
    num_trainable_params = 4



