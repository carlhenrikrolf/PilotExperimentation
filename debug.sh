config=polarisation_0.json
agent=noshieldingnopruning

out_dir=debug
rm -f -r results/$out_dir
python3 -m debugpy --listen 5678 train.py $agent $config $out_dir