input=debug

dir=input
config=input
rm -fr results/$dir
python3 -m debugpy --listen 5678 train.py $config
