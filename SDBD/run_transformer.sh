

# for file in $(ls -d ./model_checkpoint/*.pth) 
# do
#     python3 transformer.py --checkpoint ${file}
# done

# CHECKPOINT_PATH='./model_checkpoint/epochs50-7.507-2023-02-23-22:30:03.pth'
# CHECKPOINT_PATH='./model_checkpoint/epochs30-7.669-2023-02-23-22:13:54.pth'

# CHECKPOINT_PATH='./model_checkpoint/epochs30-6.934-2023-02-22-12:05:46.pth'
# CHECKPOINT_PATH='./model_checkpoint/epochs30-1.851-2023-02-28-21:28:02.pth'
# CHECKPOINT_PATH='./model_checkpoint/epochs30-4.9-2023-02-28-21:38:22.pth'


# python3 transformer.py --checkpoint ${CHECKPOINT_PATH} --lr 0.001 --bs 64 --decay 0.0001 --n_epoch 100 --gpu 3

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch 100 --gpu 1
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-24.7-2023-03-13-21:42:50.pth'
# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-24.61-2023-03-13-22:40:36.pth'
# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-12.65-2023-03-20-19:03:41.pth' #A_loss
# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-11.9-2023-03-20-19:08:20.pth' #s_loss&d_loss
# SAVE_PATH='BD_4_11'
# EPOCH=100
# GPU=2

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_BP.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-24.61-2023-03-13-22:40:36.pth'
# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs20-24.76-2023-03-22-10:09:37.pth'
# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-24.7-2023-03-13-21:42:50.pth'
# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-12.58-2023-03-20-19:19:11.pth' #A_loss
# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-11.9-2023-03-20-19:24:41.pth'  #s_loss&d_loss
# SAVE_PATH='MDD_4_11'
# EPOCH=100
# GPU=3

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_MDD.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done












# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-24.65-2023-04-17-23:09:11.pth'
# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-24.65-2023-04-19-14:16:27.pth'
# CHECKPOINT_PATH='./model_checkpoint/BP-epochs25-24.73-2023-04-21-20:34:27.pth'
# SAVE_PATH='BD_4_17'
# EPOCH=100
# GPU=2

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_BP.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/BP-epochs20-24.97-2023-04-18-22:38:30.pth'
# CHECKPOINT_PATH='./model_checkpoint/BP-epochs35-24.58-2023-04-21-20:39:50.pth'

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_BP.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_MDD.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-12.58-2023-04-17-23:21:22.pth' #A_loss

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_BP.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/BP-epochs30-11.9-2023-04-17-23:25:37.pth' #s_loss&d_loss

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_BP.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-24.74-2023-04-17-23:14:04.pth'
# SAVE_PATH='MDD_4_17'
# EPOCH=100
# GPU=3

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_MDD.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-12.59-2023-04-17-23:30:20.pth' #A_loss

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_MDD.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# CHECKPOINT_PATH='./model_checkpoint/MDD-epochs30-11.9-2023-04-17-23:35:20.pth' #s_loss&d_loss

# for lr in 0.005 0.001 0.0005
# do
#     for bs in 32 64
#     do
#         for decay in 0.01 0.001 0.0001
#         do
#             python3 transformer_MDD.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --decay $decay --n_epoch $EPOCH --gpu $GPU --save $SAVE_PATH
#         done
#     done 
# done

# SAVE_PATH='ABIDE_5_4'

# for file in ./model_checkpoint/ABIDE/*
#     do
#     if test -f $file
#     then
#         for lr in 0.005 0.001 0.0005
#         do
#             for bs in 64
#             do
#                 python3 transformer.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 100 --decay 0.001 --gpu 3 --save $SAVE_PATH
#             done
#         done
#     fi
#     done
# done

# python3 transformer.py --checkpoint './model_checkpoint/ABIDE/epochs30-a0.1-b0.5-2.665-2023-05-06-11:22:34.pth' --lr 0.005 --bs 64 --decay 0.001 --n_epoch 100 --gpu 3 --save $SAVE_PATH

# SAVE_PATH='ABIDE_5_6'

# for file in ./model_checkpoint/ABID2/*
#     do
#     if test -f $file
#     then
#         for lr in 0.005 0.001 0.0005
#         do
#             for bs in 64
#             do
#                 python3 transformer.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 100 --decay 0.001 --gpu 0 --save $SAVE_PATH
#             done
#         done
#     fi
#     done
# done


# SAVE_PATH='BD_5_7'

# for file in ./model_checkpoint/BD/*
#     do
#     if test -f $file
#     then
#         for lr in 0.005 0.001 0.0005
#         do
#             for bs in 64
#             do
#                 python3 transformer_BP.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 100 --decay 0.001 --gpu 3 --save $SAVE_PATH
#             done
#         done
#     fi
#     done
# done

# SAVE_PATH='MDD_5_7'

# for file in ./model_checkpoint/MDD/*
#     do
#     if test -f $file
#     then
#         for lr in 0.005 0.001 0.0005
#         do
#             for bs in 64
#             do
#                 python3 transformer_MDD.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 100 --decay 0.001 --gpu 3 --save $SAVE_PATH
#             done
#         done
#     fi
#     done
# done

# SAVE_PATH='ABIDE'
# FILE='./model_checkpoint/ABIDE/epochs30-a1.0-b1.0-6.934-2023-05-09-18:56:57.pth'

# for lr in 0.005 0.001 0.0005
#     do
#         for bs in 16
#         do
#             for decay in 0.01 0.1
#             do
#                 python3 transformer.py --checkpoint ${FILE} --lr $lr --bs $bs --n_epoch 100 --decay $decay --gpu 3 --save $SAVE_PATH
#             done
#         done
#     done
# done

# SAVE_PATH='ABIDE_6_1'

# for file in ./model_checkpoint/ABIDE_6_1/*
#     do
#     if test -f $file
#     then
#         for lr in 0.005 0.001 0.0005
#         do
#             for bs in 16
#             do
#                 python3 transformer.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 100 --decay 0.001 --gpu 3 --save $SAVE_PATH
#             done
#         done
#     fi
#     done
# done

# FILE='./model_checkpoint/epochs30-6.934-2023-02-22-12:05:46.pth'
# SAVE_PATH='ABIDE_6_4'

# python3 transformer.py --checkpoint $FILE --lr 0.005 --bs 64 --n_epoch 100 --decay 0.001 --gpu 3 --save $SAVE_PATH

# SAVE_PATH='ABIDE_6_16_2'

# for file in ./model_checkpoint/ABIDE_aal/*
#     do
#     if test -f $file
#     then
#         for lr in 0.0005 0.001
#         do
#             for bs in 32 64
#             do
#                 for decay in 0.01
#                 do
#                 python3 transformer_ABIDE_aal.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 100 --decay $decay --gpu 2 --save $SAVE_PATH
#                 done
#             done
#         done
#     fi
#     done
# done

# SAVE_PATH='ABIDE_Transformer_head'
# CHECKPOINT_PATH='./model_checkpoint/epochs30-6.934-2023-02-22-12:05:46.pth'

# for head in 1 2 4 8 16
# do
#     for lr in 0.001 0.0005
#     do
#         for bs in 32 64
#         do
#         python3 transformer.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --n_epoch 100 --decay 0.001 --head $head --gpu 3 --save $SAVE_PATH
#         done
#     done
# done

# SAVE_PATH='ABIDE_8_2'

# for file in ./model_checkpoint/ABIDE_aal_5/*
#     do
#     if test -f $file
#     then
#         for lr in 0.0005 0.001 0.005
#         do
#             for bs in 128 64
#             do
#                 for decay in 0.01 0.001 0.0001
#                 do
#                 python3 transformer_ABIDE_aal.py --checkpoint ${file} --lr $lr --bs $bs --n_epoch 200 --decay $decay --gpu 0 --save $SAVE_PATH
#                 done
#             done
#         done
#     fi
#     done
# done


SAVE_PATH='ABIDE_8_17'
CHECKPOINT_PATH='/home/fennel/Brain/model_checkpoint/ABIDE2/epochs30-a1.0-b1.0-6.916-2023-08-17-22:36:08.pth'
# CHECKPOINT_PATH='/home/fennel/Brain/model_checkpoint/ABIDE2/epochs20-a1.0-b1.0-7.048-2023-08-19-20:13:31.pth'

for lr in 0.001 0.0005
do
    for bs in 64 32
    do
        for decay in 0.001 0.0001
        do
        python3 transformer.py --checkpoint ${CHECKPOINT_PATH} --lr $lr --bs $bs --n_epoch 300 --decay $decay --gpu 0 --save $SAVE_PATH
        done
    done
done