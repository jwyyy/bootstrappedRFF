# Experiment 1 --- S-Curve -- operator norm
python3 bootstrap.py --name exp1_sig05_op --problem example1 --n 20000 --gamma 0.5 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

python3 bootstrap.py --name exp1_sig1_op --problem example1 --n 20000 --gamma 1.0 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

python3 bootstrap.py --name exp1_sig4_op --problem example1 --n 20000 --gamma 4.0 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000


# Experiment 2 --- Lorenz System
python3 bootstrap.py --name exp2_sig05_op --problem example2 --n 25000 --gamma 0.5 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

python3 bootstrap.py --name exp2_sig1_op --problem example2 --n 25000 --gamma 1.0 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

python3 bootstrap.py --name exp2_sig4_op --problem example2 --n 25000 --gamma 4.0 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

# MNIST

python3 bootstrap.py --name exp3_sig1_op --problem mnist --n 50000 --gamma 0.5 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

python3 bootstrap.py --name exp3_sig4_op --problem mnist --n 50000 --gamma 1.0 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000

python3 bootstrap.py --name exp3_sig6_op --problem mnist --n 50000 --gamma 4.0 --runs 500 --boots 30 --samples 50 100 200 400 700 1000 1500 2000 4000 6000


