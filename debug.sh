out_dir=peucrl_minus_r_minus_safety_10
rm -f -r results/.$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r_minus_safety polarisation_11.json $out_dir