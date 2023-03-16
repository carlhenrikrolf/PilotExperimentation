out_dir=peucrl_minus_r_minus_action_pruning_8
rm -f -r results/.$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r_minus_action_pruning polarisation_9.json $out_dir