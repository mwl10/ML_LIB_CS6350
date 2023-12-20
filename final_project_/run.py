from absl import app, flags
from ml_collections.config_flags import config_flags
from utils import cprint, create_path, get_logger
import os
import numpy as np
from dataset import (NumpyLoader,
                     BluesDriverDataset)
from jax import random
import jax.numpy as jnp
from jax import default_backend
from models.models import NOpModelNd
from models.mutils import (UnitGaussianNormalizer,
                           get_grid, 
                           create_lr_schedule_fn)

from flax.training.train_state import TrainState
import optax
from jax import jit, value_and_grad
from utils import (quad_type_mapping, 
                   param_init_mapping, 
                   non_linearity_mapping,
                   kernel_mapping)

import orbax
from flax.training import orbax_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Work directory.")
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string('expsig', None, 'exp signature')
flags.mark_flags_as_required(["workdir", "config", 'expsig'])


##########################################################

def main(argv):
    workdir = FLAGS.workdir
    config = FLAGS.config
    expsig = FLAGS.expsig

    # exp_signature = f'{int(SystemRandom().random() * 100000)}'
    create_path(workdir, verbose=False)
    
    exp_workdir = os.path.join(workdir, expsig)
    create_path(exp_workdir, verbose=False)
    
    logger = get_logger(
        os.path.join(exp_workdir, 'log.txt'),
        displaying=False
    )
    
    logger.info('=============== Experiment Setup ===============')
    logger.info(config)
    logger.info('================================================')

    logger.info('loading data...')
    
    logger.info(f'DEVICE = {default_backend()}')

    if config.dataset.lower() == 'blues_driver':
        data = BluesDriverDataset(config)
    
    train = data.get_train_data()
    test = data.get_test_data()
    logger.info(f'created datasets, {train.x.shape=}, {train.y.shape=}')
    
    ### training 
    key = random.PRNGKey(seed=config.seed)
    nx = train.x.shape[1]

    if config.data.normalize:
        x_normalizer = UnitGaussianNormalizer(train.x)
        y_normalizer = UnitGaussianNormalizer(train.y)  
        train.x = x_normalizer.encode(train.x)  
        test.x = x_normalizer.encode(test.x)   

    train_loader = NumpyLoader(train,
                               batch_size=config.training.batch_size,
                               shuffle=config.training.shuffle)
    test_loader = NumpyLoader(test,
                              batch_size=config.testing.batch_size) 
   
    # map from strings in config to objects (like a psd_kernel) for the model to digest
    m = config.model
    interpolation_kernel = kernel_mapping[m.interpolation_kernel]
    integration_kernels = tuple(kernel_mapping[kernel] for kernel in m.integration_kernels.rstrip(',').split(','))
    non_linearity = non_linearity_mapping[m.non_linearity]
    dense_init = param_init_mapping[m.dense_init]
    kernel_init = param_init_mapping[m.kernel_init]
    quadrature_fn = quad_type_mapping[m.quad_type]
    
    model = NOpModelNd(quadrature_fn=quadrature_fn,
                       integration_kernels=integration_kernels,
                       interpolation_kernel=interpolation_kernel,
                       non_linearity=non_linearity,
                       dense_init=dense_init,
                       kernel_init=kernel_init,
                       num_quad_pts=m.num_quad_pts,
                       width=m.width,
                       residual_block=m.residual_block,
                       channel_lift=m.channel_lift,
                       channel_lift_size=m.channel_lift_size,
                       num_output_layers=m.num_output_layers,
                       conv=m.conv,
                       secs_per_fn=config.data.secs_per_fn
                       )
    
    ### initialize model
    f_x = train.x[:1] ### for model init
    s = get_grid(f_x.shape[1:],secs_per_fn=config.data.secs_per_fn)
    variables = model.init(key,f_x,s)
    params = variables['params']
    num_epochs=config.training.epochs
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs

    opt= config.optim
    ### cos annealing params
    lr_schedule_fn = create_lr_schedule_fn(total_steps=total_steps,
                                           init_value=opt.init_value, 
                                           warmup_frac=opt.warmup_frac,
                                           peak_value=opt.peak_value,
                                           end_value=opt.end_value,
                                           num_cycles=opt.num_cycles,
                                           gamma=opt.gamma)
                                        
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=optax.adam(lr_schedule_fn)
                             )
    
    def l2_relative_error(params, data, labels, reduce=None):
        f_x = data
        preds = state.apply_fn({'params': params}, f_x, s)
        if config.data.normalize:
            preds = y_normalizer.decode(preds)  
        num_examples = f_x.shape[0]
        p_norm = lambda x, ord, dim: jnp.sum(jnp.abs(x)**ord, axis=dim)**(1./ord)
        diff_norms = p_norm(preds.reshape(num_examples,-1) - labels.reshape(num_examples,-1), ord=2, dim=1)
        label_norms = p_norm(labels.reshape(num_examples, -1), ord=2,dim=1)
        losses = diff_norms/label_norms
        if reduce=='sum': return jnp.sum(losses) ### sum across the batch 
        if reduce=='mean': return jnp.mean(losses) 
        else: return losses

    def mse(params, data,labels, reduce=None):
        f_x = data
        preds = state.apply_fn({'params': params}, f_x, s)
        if config.data.normalize:
            preds = y_normalizer.decode(preds)  
        num_examples = f_x.shape[0]
        mse = lambda x,y,dim: ((x - y)**2).mean(axis=dim)
        losses = mse(preds.reshape(num_examples,-1),
                     labels.reshape(num_examples,-1),dim=1)
        if reduce=='sum': return jnp.sum(losses)
        if reduce=='mean': return jnp.mean(losses)
        else: return losses

    if config.training.loss_fn == 'mse': loss_fn = mse
    else: loss_fn = l2_relative_error

    @jit
    def train_step(state: TrainState, batch):
        data, labels = batch 
        loss_value, grads = value_and_grad(loss_fn)(state.params, data, labels, reduce='mean')
        state = state.apply_gradients(grads=grads)
        metrics = loss_value
        return state, metrics
    
    def train_one_epoch(state: TrainState, dataloader):
        batch_metrics = jnp.zeros((len(dataloader),))
        for i,batch in enumerate(dataloader):
            state, batch_metric = train_step(state, batch)
            batch_metrics = batch_metrics.at[i].set(batch_metric)
        epoch_metrics = batch_metrics.mean()
        return state, epoch_metrics

    @jit
    def eval_step(state, batch): 
        data, labels = batch
        batch_metric = l2_relative_error(state.params, data, labels, reduce='sum')
        return batch_metric ## state ofc not updated in eval
    
    def eval_model(state, dataloader):
        batch_metrics = jnp.zeros((len(dataloader),))
        for i,batch in enumerate(dataloader):
            batch_metric = eval_step(state, batch)
            batch_metrics = batch_metrics.at[i].set(batch_metric)
        epoch_metrics = batch_metrics.sum()
        return epoch_metrics.item()

    losses = np.zeros((num_epochs,2)) # train/test losses
    
    for epoch in range(num_epochs):
        state, epoch_metrics = train_one_epoch(state, train_loader)
        losses[epoch,0] = epoch_metrics
        
        if epoch % config.training.log_at == 0:
            logger.info(f'step {epoch}, train_loss: {epoch_metrics}')  
        if epoch % config.testing.eval_at == 0:
            test_loss = eval_model(state, test_loader) / len(test.x)
            losses[epoch,1] = test_loss 
            logger.info(f'step {epoch}, test_loss: {test_loss}')
            
    ##### checkpointing !
    ckpt = {'model': state, 
            'config': config.to_dict(), 
            'losses': losses,
            'step': epoch}

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(os.path.join(exp_workdir,'ckpt'),
                            ckpt, 
                            save_args=save_args,
                            force=True)
    

if __name__ == "__main__":
    app.run(main)