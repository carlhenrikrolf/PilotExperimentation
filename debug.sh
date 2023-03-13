out_dir=peucrl_minus_r_11
rm -f -r results/.$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r polarisation_10.json $out_dir 

out_dir=peucrl_minus_r_minus_shield_9
rm -f -r results/.$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r_minus_shield polarisation_10.json $out_dir 

out_dir=peucrl_minus_r_minus_action_pruning_5
rm -f -r results/.$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r_minus_action_pruning polarisation_10.json $out_dir 

out_dir=peucrl_minus_r_minus_safety_6
rm -f -r results/.$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r_minus_safety polarisation_10.json $out_dir 