#!/bin/bash

cd /develop/graphstorm
DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFERs=2
export PYTHONPATH=$GS_HOME/python/
echo "127.0.0.1" > ip_list.txt
mkdir -p /tmp/ogbn-arxiv-nc
cp ip_list.txt /tmp/ogbn-arxiv-nc


# commands from GLEM repo: https://github.com/AndyJZhao/GLEM/tree/main/OGB/ogbn-arxiv
# arxiv FtV1 Roberta-large
# --lr=1e-05 --eq_batch_size=36 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.1 --cla_dropout=0.1 --cla_bias=T --epochs=4 --warmup_epochs=0.2 --eval_patience=5000

python src/models/GLEM/trainGLEM.py --dataset=arxiv_TA --em_order=LM-first --gnn_ckpt=RevGAT --gnn_early_stop=300 --gnn_epochs=2000 --gnn_input_norm=T --gnn_label_input=F --gnn_model=RevGAT --gnn_pl_ratio=1 --gnn_pl_weight=0.05 --inf_n_epochs=2 --inf_tr_n_nodes=100000 --lm_ce_reduction=mean --lm_cla_dropout=0.4 --lm_epochs=3 --lm_eq_batch_size=30 --lm_eval_patience=30460 --lm_init_ckpt=None --lm_label_smoothing_factor=0 --lm_load_best_model_at_end=T --lm_lr=2e-05 --lm_model=Deberta --lm_pl_ratio=1 --lm_pl_weight=0.8 --pseudo_temp=0.2  --gpus=0 

# BERT ft
python3 -m graphstorm.run.gs_node_classification \
            --workspace /tmp/ogbn-arxiv-nc \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /data/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc_lm.yml \
            --save-model-path /data/ogbn-arxiv-nc/models/bert_only \
            --save-perf-results-path /data/ogbn-arxiv-nc/models/bert_only \
            --lm-encoder-only \
            --batch-size 32 \
            --lr 0.00002 \
            --lm-tune-lr 0.00002 \
            --num-epochs 10

### with weight decay
python3 -m graphstorm.run.gs_node_classification \
            --workspace /tmp/ogbn-arxiv-nc \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /data/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc_lm.yml \
            --save-model-path /data/ogbn-arxiv-nc/models/bert_only \
            --save-perf-results-path /data/ogbn-arxiv-nc/models/bert_only \
            --lm-encoder-only \
            --batch-size 36 \
            --lr 0.00001 \
            --lm-tune-lr 0.00001 \
            --lm-train-nodes 36 \
            --num-epochs 10 \
            --wd-l2norm 0.01


### with scheduler
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /data/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc_lm.yml \
            --save-model-path /data/ogbn-arxiv-nc/models/bert_only \
            --save-perf-results-path /data/ogbn-arxiv-nc/models/bert_only \
            --lm-encoder-only \
            --batch-size 36 \
            --lr 0.00001 \
            --lm-tune-lr 0.00001 \
            --lm-train-nodes 36 \
            --num-epochs 10 \
            --wd-l2norm 0.01 \
            --warmup-epochs 0.2

# GLEM semi-sup
python3 -m graphstorm.run.gs_node_classification \
            --workspace /tmp/ogbn-arxiv-nc \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /data/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc_glem.yml \
            --save-model-path /data/ogbn_arxiv_nc_train_val_1p_8t/glem/models \
            --save-perf-results-path /data/ogbn_arxiv_nc_train_val_1p_8t/glem/models \
            --num-epochs 33 \
            --freeze-lm-encoder-epochs 30 \
            --lm-tune-lr 0.00002 \
            --lr 0.00002 --batch-size 32 \
            --semi-supervised true

python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /data/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc_glem.yml \
            --save-model-path /data/ogbn_arxiv_nc_train_val_1p_8t/glem_nofreeze/models \
            --save-perf-results-path /data/ogbn_arxiv_nc_train_val_1p_8t/glem_nofreeze/models \
            --num-epochs 10 \
            --lm-tune-lr 0.00002 \
            --lr 0.00002 --batch-size 32 \
            --semi-supervised true

### Load pre-trained -> not supported
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /data/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc_glem.yml \
            --save-model-path /data/ogbn_arxiv_nc_train_val_1p_8t/glem_warmstart/models \
            --save-perf-results-path /data/ogbn_arxiv_nc_train_val_1p_8t/glem_warmstart/models \
            --num-epochs 3 \
            --lm-tune-lr 0.00002 \
            --lr 0.00002 --batch-size 32 \
            --semi-supervised true \
            --restore-model-path /data/ogbn-arxiv-nc/models/bert_only/epoch-9 \
            --restore-optimizer-path /data/ogbn-arxiv-nc/models/bert_only/epoch-9
