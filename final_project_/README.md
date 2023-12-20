### ML 6350 Final Project codebase

Some relevant files are: 

[dataset.py](dataset.py) and 
[audio_data_tinker.ipynb](./dev_notebooks/audio_data_tinker.ipynb) for preprocessing the audio data and preparing it for training! 

For some results, take a look at [blues_driver_results.ipynb](blues_driver_results.ipynb).



### Some details:

#### [kernels.py](kernels.py)
I'm using kernel implementations from tfp psd_kernels, more specifically, the jax substrate (tensorflow_probability.substrates.jax.math.psd_kernels). I subclass them to include an additional num_trainable_params class attribute, which is used initialize the model parameters. 

#### [run.py](run.py)
Model training code mostly. Examples of calling run.py are in ./deploy_scripts, but for instance

```bash
exp="1"
dataset="blues_driver"
ktype=('Gaussian' 'Wendland')
for k in ${ktype[@]}; do
    for numk in $(seq 1 5); do
    echo "kernel type, ${k} w/ ${numk} int blocks"
    sublist=$(printf "${k},%.0s" $(seq 1 ${numk}))
    python3 run.py --workdir=./exps/${dataset}/${exp} --config=./config/${dataset}.py \
    --expsig="${k}_${numk}" --config.model.integration_kernels=${sublist[@]}
    done
done
```

This will save logs, the config_dict and a np array of the train/test losses for each run call to the folder workdir/expsig/, (something like './exps/blues_driver/1/Gaussian_1'). You can override the default config, taken from config/darcy.py here, w/ something like '--config.model.___='. "workdir", "config" & "expsig" are required. 


#### [dataset.py](dataset.py)
Jax/flax don't include dataloaders/dataclasses, so we adapt the torch ones here. There is a class for the darcy,burgers/adv2 benchmarks,which should should clarify how to use them from the raw files in the google drive. 


#### [models](models/models.py)
The model implementation exists at models/models.py (it needs to be commented out better), but generally works by taking in a list of the above psd_kernels to define the architecture/integration-block layers of the Quadrature-operator model. Something like [Gaussian, Wendland, MaternOneHalf].


#### [quadratures.py](quadratures.py)
The model is initialized with a given quadrature rule from numpy, i.e.
numpy.polynomial.chebyshev.chebgauss, np.polynomial.hermite.hermgauss, I have an extra one in this file for a Gauss-Lobatto quadrature (gLLNodesAndWeights).