from jax import numpy as jnp
import optax

### cos annealing 
def create_lr_schedule_fn(total_steps,
                     init_value=1e-5,
                     warmup_frac=0.3, 
                     peak_value=3e-5, 
                     end_value=1e-5,
                     num_cycles=6,
                     gamma=0.85
                     ):
    decay_steps = total_steps / num_cycles
    schedules = []
    boundaries = []
    boundary=0
    for _ in range(num_cycles):
        schedule = optax.warmup_cosine_decay_schedule(init_value=init_value,
                                                warmup_steps=decay_steps * warmup_frac, 
                                                peak_value=peak_value, 
                                                decay_steps=decay_steps, 
                                                end_value=end_value,
                                                exponent=2)
        boundary = decay_steps + boundary 
        boundaries.append(boundary)
        peak_value = peak_value * gamma
        schedules.append(schedule)
    return optax.join_schedules(schedules=schedules, 
                                boundaries=boundaries)


def get_grid(resolutions: list[int], secs_per_fn=1):  
    ndims = len(resolutions)
    grid = [jnp.linspace(0,secs_per_fn,res) for res in resolutions]
    grid = jnp.array(jnp.meshgrid(*grid))
    grid = grid.transpose(list(range(1,len(grid)+1)) + [0])
    return grid ### i.e. nx,1 or nx,ny,nz,3 or nx,ny,2 

param_count = lambda params : sum(x.size for x in jax.tree_leaves(params))


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = jnp.mean(x, axis=0)
        self.std = jnp.std(x, axis=0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
        # x is in shape of batch*(spatial discretization size) 
        # or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x
    

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

            