import ml_collections

def get_default_configs():

    config = ml_collections.ConfigDict()

    # data preprocessing
    config.data = data = ml_collections.ConfigDict()
    data.normalize = True
    data.ntrain_frac = 0.5
    data.ntest_frac = data.ntrain_frac
    data.downsample = True
    data.secs_per_fn = 0.1


    # model init
    config.model = model = ml_collections.ConfigDict()
    model.width = 128
    model.integration_kernels = ('Gaussian,') * 2
    model.interpolation_kernel = 'Gaussian'
    model.non_linearity = 'gelu'
    model.dense_init = 'he_normal'
    model.quad_type = 'hermite'
    model.num_quad_pts = 10 
    model.kernel_init = 'constant'
    model.residual_block = True
    model.channel_lift = True
    model.channel_lift_size=128
    model.num_output_layers = 1
    model.conv = True
    
    # training
    config.training = training = ml_collections.ConfigDict()
    training.loss_fn = 'l2_relative_error' ### or 'mse'
    training.batch_size = 20
    training.epochs = 15000
    training.log_at = 20
    training.shuffle = True


    # testing
    config.testing = testing = ml_collections.ConfigDict()
    testing.batch_size = 20
    testing.eval_at = 100
    # testing.samples = 20

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    ### cos annealing params
    optim.init_value=1e-5 
    optim.warmup_frac=0.3
    optim.peak_value=3e-5 
    optim.end_value=1e-5 
    optim.num_cycles=6 
    optim.gamma=0.85

    # logging
    config.logging = logging =  ml_collections.ConfigDict()
    logging.display = False

    # misc
    config.seed = 42
    config.dataset='blues_driver'

    return config


def get_config():
    config = get_default_configs()
    data = config.data
    data.x_train_fp = 'datasets/blues_driver/clean_inmyarmsriff.mp3'
    data.y_train_fp = 'datasets/blues_driver/dirty_inmyarmsriff.mp3'
    data.x_test_fp = 'datasets/blues_driver/clean_inmyarmsriff.mp3'
    data.y_test_fp = 'datasets/blues_driver/dirty_inmyarmsriff.mp3'
    return config

