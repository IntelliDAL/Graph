# GPU=3

# for a in 0.1 0.2 0.3 0.4 0.5
#     do
#     for b in 0.9 0.8 0.7 0.6 0.5
#         do
#         python3 disentangle.py --a ${a} --b ${b} --gpu ${GPU}
#     done
# done 

# python3 disentangle.py --a 0.1 --b 0.9 --gpu ${GPU}

# GPU=3

# for a in 0.1 0.2 0.3 0.4 0.5
#     do
#     for b in 0.9 0.8 0.7 0.6 0.5
#         do
#         python3 disentangle_BP.py --a ${a} --b ${b} --gpu ${GPU}
#     done
# done 

# GPU=2

# for a in 0.1 0.2 0.3 0.4 0.5
#     do
#     for b in 0.9 0.8 0.7 0.6 0.5
#         do
#         python3 disentangle_MDD.py --a ${a} --b ${b} --gpu ${GPU}
#     done
# done 

# python3 disentangle.py --a 1.0 --b 1.0 --gpu 3

GPU=0

for epochs in 20
    do
    python3 disentangle.py --a 1.0 --b 1.0 --gpu ${GPU} --n_epoch ${epochs}
done
