

python3 $GS_HOME/examples/acm_data.py \
    --output-path /data/acm_raw_featureless_author \
    --output-type raw_w_text_featureless_author

python3 -m graphstorm.gconstruct.construct_graph \
    --conf-file /data/acm_raw_featureless_author/config.json \
    --output-dir /data/acm_nc_w_text_fl_author \
    --num-parts 1 \
    --graph-name acm

# train LM-GNN
python3 -m graphstorm.run.gs_node_classification \
        --part-config /data/acm_nc_w_text_fl_author/acm.json \
        --ip-config ip_list.txt \
        --num-trainers 2 \
        --num-servers 1 \
        --num-samplers 0 \
        --ssh-port 2222 \
        --cf $GS_HOME/examples/use_your_own_data/acm_lm_nc_fl_author.yaml \
        --num-epochs 3 
        # --node-feat-name paper:feat subject:feat
        

# train GLEM
python3 -m graphstorm.run.gs_node_classification \
        --part-config /data/acm_nc_w_text_fl_author/acm.json \
        --ip-config ip_list.txt \
        --num-trainers 2 \
        --num-servers 1 \
        --num-samplers 0 \
        --ssh-port 2222 \
        --cf $GS_HOME/examples/use_your_own_data/acm_glem_nc_fl_author.yaml \
        --num-epochs 3 \
        --use-pseudolabel true

python3 -m graphstorm.run.gs_node_classification \
        --part-config /data/acm_nc_w_text_fl_author/acm.json \
        --ip-config ip_list.txt \
        --num-trainers 1 \
        --num-servers 1 \
        --num-samplers 0 \
        --ssh-port 2222 \
        --cf $GS_HOME/examples/use_your_own_data/acm_glem_nc_fl_author.yaml \
        --num-epochs 3
