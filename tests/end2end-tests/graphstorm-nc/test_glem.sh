#!/bin/bash

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_np

echo "127.0.0.1" > ip_list.txt

cat ip_list.txt


echo "**************dataset: MovieLens classification, RGCN layer: 1, node feat: BERT nodes: movie, user inference: mini-batch save model save emb node"
# python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_text_train_val_1p_4t/movie-lens-100k-text.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_utext.yaml  --save-model-path /data/gsgnn_nc_ml_text/ --topk-model-to-save 1 --save-embed-path /data/gsgnn_nc_ml_text/emb/ --num-epochs 3 | tee train_log.txt

# run gsgnn_np.py directly:
python3 -m torch.distributed.launch -m graphstorm.run.gsgnn_np.gsgnn_np --part-config /data/movielen_100k_text_train_val_1p_4t/movie-lens-100k-text.json --cf ml_nc_utext.yaml  --save-model-path /data/gsgnn_nc_ml_text/ --topk-model-to-save 1 --save-embed-path /data/gsgnn_nc_ml_text/emb/ --num-epochs 3 | tee train_log.txt
# Perf:
# best_test_score: {'accuracy': 0.3076923076923077}
# best_val_score: {'accuracy': 0.2976190476190476}

# test glem
python3 -m torch.distributed.launch -m graphstorm.run.gsgnn_np.gsgnn_np --part-config /data/movielen_100k_text_train_val_1p_4t/movie-lens-100k-text.json --cf ml_nc_utext_glem.yml  --save-model-path /data/gsgnn_nc_ml_text/ --topk-model-to-save 1 --num-epochs 3 | tee train_log.txt
# Perf:
# best_test_score: {'accuracy': 0.3136094674556213}
# best_val_score: {'accuracy': 0.2976190476190476}


# Test on OGB-arxiv:
## 1. prepare OGB datasets with texts
## Using for tokenizer bert-base-uncased
cd /develop/graphstrom
# same with partition_graph.py, need to have --retain-original-features false
# python3 tools/gen_ogb_dataset.py \
#     --dataset ogbn-arxiv \
#     --filepath /data/dataset/ogbn_arxiv/ \
#     --savepath /tmp/ogbn-arxiv-nc/ \
#     --retain-original-features false

python3 tools/partition_graph.py \
    --dataset ogbn-arxiv \
    --retain-original-features false \
    --filepath /data/dataset/ogbn_arxiv/ \
    --num-parts 1 \
    --num-trainers-per-machine 8 \
    --output /tmp/ogbn_arxiv_nc_train_val_1p_8t


touch /tmp/ogbn-arxiv-nc/ip_list.txt
echo 127.0.0.1 > /tmp/ogbn-arxiv-nc/ip_list.txt

## LM GNN co-training without GLEM:
python3  -m graphstorm.run.gs_node_classification \
            --workspace /tmp/ogbn-arxiv-nc \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /tmp/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
            --save-model-path /tmp/ogbn-arxiv-nc/models \
            --save-perf-results-path /tmp/ogbn-arxiv-nc/models
# bs=1024
# Part 0 | Epoch 00000 | Batch 000 | Loss: 18.5131 | Time: 288.0287
# Epoch 0 take 3240.554735660553
# computing GNN embeddings: 185.3919 seconds
# Step 12 | Validation accuracy: 0.0766
# Step 12 | Test accuracy: 0.0588
# Step 12 | Best Validation accuracy: 0.0766
# Step 12 | Best Test accuracy: 0.0588
# Step 12 | Best Iteration accuracy: 12.0000
#  Eval time: 185.5877, Evaluation step: 12.0000
# successfully save the model to /tmp/ogbn-arxiv-nc/models/epoch-0            

python3  -m graphstorm.run.gs_node_classification \
            --workspace /tmp/ogbn-arxiv-nc \
            --num-trainers 8 \
            --num-servers 1 \
            --num-samplers 0 \
            --part-config /tmp/ogbn_arxiv_nc_train_val_1p_8t/ogbn-arxiv.json \
            --ip-config  /tmp/ogbn-arxiv-nc/ip_list.txt \
            --ssh-port 2222 \
            --cf /develop/graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
            --save-model-path /tmp/ogbn-arxiv-nc/models \
            --save-perf-results-path /tmp/ogbn-arxiv-nc/models \
            --batch-size 10240

# Part 0 | Epoch 00000 | Batch 000 | Loss: 21.2182 | Time: 706.9124
# Epoch 0 take 1008.7581975460052
# computing GNN embeddings: 186.0783 seconds
# Step 2 | Validation accuracy: 0.0349
# Step 2 | Test accuracy: 0.0388
# Step 2 | Best Validation accuracy: 0.0349
# Step 2 | Best Test accuracy: 0.0388
# Step 2 | Best Iteration accuracy: 2.0000
#  Eval time: 186.1285, Evaluation step: 2.0000
# successfully save the model to /tmp/ogbn-arxiv-nc/models/epoch-0

## Without LM:
# best_test_score: {'accuracy': 0.6486019381519659}
# best_val_score: {'accuracy': 0.6670693647437833}





# test glem + inference



# pylint --rcfile=$GS_HOME/tests/lint/pylintrc $GS_HOME/python/graphstorm/model/node_glem.py