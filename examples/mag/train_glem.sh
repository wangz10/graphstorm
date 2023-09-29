#!/bin/bash

## Set up docker:

# Start a docker, mapping dev code bases into its volume
# and set env vars
nvidia-docker run \
    -v /mnt/efs/gsf-data:/data \
    -v /home/ubuntu/graphstorm:/develop/graphstorm \
    --network=host \
    --shm-size=700g \
    --env DGL_HOME=/root/dgl \
    --env GS_HOME=/develop/graphstorm \
    --env PYTHONPATH=/develop/graphstorm/python/ \
    -d graphstorm:0908

cd $GS_HOME
# DGL_HOME=/root/dgl
# GS_HOME=$(pwd)
# export GS_HOME=/develop/graphstorm
# export PYTHONPATH=$GS_HOME/python/
echo "127.0.0.1" > ip_list.txt

python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_glem_nc.yaml --save-model-path /data/mag_min_1part3/glem_nc/ --topk-model-to-save 1 --num-epochs 10 --use-pseudolabel true

# continue from checkpoint
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_glem_nc.yaml --save-model-path /data/mag_min_1part3/glem_nc/ --topk-model-to-save 1 --num-epochs 10 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/glem_nc/epoch-8/


python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt2/ --topk-model-to-save 1 --num-epochs 10 --use-pseudolabel true
# continue from checkpoint
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc.yaml  --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt2/ --topk-model-to-save 1 --num-epochs 20 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt2/epoch-9/

python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc2.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt1/ --topk-model-to-save 1 --num-epochs 10 --use-pseudolabel true


# larger LR, more pt epochs:
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc3.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt5/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true
# run inference 
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc3.yaml --inference --restore-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt5/epoch-14

## GLEM model inference with gnn
python3 -m graphstorm.run.gs_node_classification --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc_inf_w_gnn.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt5_inf_w_gnn/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true

## --- Load Da's BERT checkpoint ---
## Only use the BERT's `model.bin` from Da, the rest of the files are from the best GLEM checkpoint in `glem_nc_lmfirst_pt2`
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc3.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt5_from_bert_lp_ft/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_lp_model/

# also load Da's GNN, 
# before running this, create a softlink to the trained GNN checkpoint
# ln -s /data/mag_gnn_model/epoch-7/ /data/mag_min_1part3/mag_bert_lp_model/GNN
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_glem_from_checkpoints.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt0_from_bert_lp_ft_and_gnn/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_lp_model/ 

# Use Da's BERT fine-tuned for nc 
# ln -s /data/mag_bert_models/mag_bert_nc_model/epoch-9/ /data/mag_min_1part3/mag_bert_nc_model/LM
# ln -s /data/mag_gnn_model/epoch-7/ /data/mag_min_1part3/mag_bert_nc_model/GNN
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_glem_from_checkpoints.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt5_from_bert_nc_ft_and_gnn/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/ 

python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_glem_from_checkpoints.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt0_from_bert_nc_ft_and_gnn/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/ 

python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_glem_nc.yaml --save-model-path /data/mag_min_1part3/glem_nc_lmfirst_pt0_from_bert_nc_ft_and_gnn/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/ --restore-model-layers embed

# 4 parts:
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p.yaml --save-model-path /data/mag_min_4parts/glem_nc_lmfirst_pt5_from_bert_nc_ft_and_gnn/ --topk-model-to-save 1 --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/ --save-perf-results-path /data/mag_min_4parts/glem_nc_lmfirst_pt5_from_bert_nc_ft_and_gnn

# on `glem_full_graph_inference` branch:
# training_method:
#   name: glem
#   kwargs:
#     em_order_gnn_first: false
#     inference_using_gnn: true
#     pl_weight: 0.5
#     num_pretrain_epochs: 10
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p_gnn_inf.yaml --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/ --restore-model-layers embed --freeze-lm-encoder-epochs 10 --batch-size 64 --lr 0.0001 --save-model-path /data/mag_min_4parts/glem_cotrain_from_pretrained_nopt_full_graph_inf --topk-model-to-save 1

# use a newly trained BERT-GNN checkpoint (acc=.37)
mkdir /data/mag_gnn_model/from_ep7/epoch-12/GLEM
ln -s /data/mag_gnn_model/from_ep7/epoch-12 /data/mag_gnn_model/from_ep7/epoch-12/GLEM/GNN
ln -s /data/mag_gnn_model/from_ep7/epoch-12 /data/mag_gnn_model/from_ep7/epoch-12/GLEM/LM

python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p_gnn_inf.yaml --num-epochs 50 --use-pseudolabel true --restore-model-path /data/mag_gnn_model/from_ep7/epoch-12/GLEM --restore-model-layers embed --freeze-lm-encoder-epochs 10 --batch-size 64 --lr 0.0001 --save-model-path /data/mag_min_4parts/glem_cotrain_from_zw_pretrained_nopt_full_graph_inf --topk-model-to-save 1


# evaluate previous trained on the `glem_full_graph_inference` branch
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p_nopt.yaml --restore-model-path /data/mag_min_4parts/glem_cotrain_from_pretrained_nopt/epoch-6 --restore-model-layers embed,decoder --inference 
## inf with LM: 0.0020
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p_nopt_gnn.yaml --restore-model-path /data/mag_min_4parts/glem_cotrain_from_pretrained_nopt/epoch-6 --restore-model-layers embed,decoder --inference 
## inf with GNN: 0.0168

# restore Da's GNN model and inferenece with GLEM gnn head
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p_nopt_gnn.yaml --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/ --restore-model-layers embed --inference --logging-level debug
## inf with GLEM.GNN:0.0309
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf /data/mag_min_4parts/mag_glem_from_checkpoints_4p_nopt.yaml --restore-model-path /data/mag_min_1part3/mag_bert_nc_model --restore-model-layers embed --inference 
## inf with GLEM.LM: 0.0025

python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 4 --num-samplers 0 --part-config /data/mag_min_4parts/mag.json --ip-config /data/ip_list_p4_zw.txt --ssh-port 2222 --cf $GS_HOME/examples/mag/mag_gnn_nc.yaml --inference --logging-level debug --use-mini-batch-infer false --restore-model-path /data/mag_gnn_model/new/epoch-17
# --restore-model-path /data/mag_gnn_model/epoch-7/
# --restore-model-path /data/mag_min_1part3/mag_bert_nc_model/GNN
# --restore-model-path /data/mag_gnn_model/new

## inf with GNN: 0.0309

## train a new GNN on the fine-tuned BERT
## ~0.38
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 --ssh-port 2222 \
            --part-config /data/mag_min_4parts/mag.json \
            --ip-config /data/ip_list_p4_zw.txt \
            --cf $GS_HOME/examples/mag/mag_gnn_nc.yaml \
            --restore-model-path /data/mag_bert_models/mag_bert_lp_model3/epoch-0-iter-79999/ \
            --restore-model-layers dense_embed \
            --save-model-path /data/mag_gnn_model/new

## ~0.38
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 --ssh-port 2222 \
            --part-config /data/mag_min_4parts/mag.json \
            --ip-config /data/ip_list_p4_zw.txt \
            --cf $GS_HOME/examples/mag/mag_gnn_nc.yaml \
            --restore-model-path /data/mag_gnn_model/epoch-7/ \
            --restore-model-layers dense_embed \
            --save-model-path /data/mag_gnn_model/from_ep7
## 0.38
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 --ssh-port 2222 \
            --part-config /data/mag_min_4parts/mag.json \
            --ip-config /data/ip_list_p4_zw.txt \
            --cf $GS_HOME/examples/mag/mag_gnn_nc.yaml \
            --restore-model-path /data/mag_bert_models/mag_bert_nc_model/epoch-9 \
            --restore-model-layers embed \
            --save-model-path /data/mag_gnn_model/from_nc_ep9 \
            --topk-model-to-save 1


## freeze the LM during pre-training epochs, only train GNN


# --- without pseudolabel
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers 8 --num-servers 1 --num-samplers 0 --part-config /data/mag_min_1part3/mag.json --ip-config ip_list.txt --ssh-port 2222 --cf /data/mag_min_1part3/mag_glem_nc.yaml --save-model-path /data/mag_min_1part3/glem_nc_nosemi/ --topk-model-to-save 1 --num-epochs 10