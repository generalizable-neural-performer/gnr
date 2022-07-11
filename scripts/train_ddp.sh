NODES=2
GPU_PER_NODES=8
GENEBODY_ROOT=put-your-root-here
MASTER=your-address-of-master-machine
COMMAND=${1:-apps/train_genebody.py --config configs/train.txt --dataroot ${GENEBODY_ROOT} --ddp}
for ((i=0; i<$NODES; ++i)) do
    python -m torch.distributed.launch \
            --master_addr ${MASTER} \
            --master_port 29501 \
            --nproc_per_node ${GPU_PER_NODES} \
            --nnodes=${NODES} \
            --node_rank=${i} \
            ${COMMAND} &
done