epochs=30000
k="Gaussian"
sublist=$(printf "${k},%.0s" $(seq 1 ${numk}))
num_layers=1
chanlift=128
width=128
exp="1" 
for norm in 'False'; do 
for k in 'Gaussian' 'Wendland' 'MaternOneHalf'; do
for numk in 2; do
    sublist=$(printf "${k},%.0s" $(seq 1 ${numk}))
    for dataset in "blues_driver"; do
    python3 run.py --workdir=./exps/blues_driver/${exp} --config=./config/blues_driver.py \
        --expsig="${k}_${dataset}" \
        --config.data.normalize=${norm} \
        --config.training.epochs=${epochs} \
        --config.model.integration_kernels=${sublist[@]} \
        --config.model.num_output_layers=${num_layers} \
        --config.model.width=${width} \
        --config.model.channel_lift_size=${chanlift}
    done
done
done
done
