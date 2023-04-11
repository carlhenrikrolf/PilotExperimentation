out_dir=debug
rm -f -r results/$out_dir
python3 -m debugpy --listen 5678 train.py peucrl_minus_r polarisation_17.json $out_dir