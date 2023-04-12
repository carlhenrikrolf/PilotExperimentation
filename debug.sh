config=polarisation_11.json
agent=ucrl2

out_dir=debug
rm -f -r results/$out_dir
python3 -m debugpy --listen 5678 train.py $agent $config $out_dir