export CUDA_VISIBAL_DEVICES=2,3
torchrun --nproc_per_node=2 train_fsdp.py > train_fsdp.log 2>&1