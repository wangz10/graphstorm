This folder contains the data processing script to process the raw MAG dataset
downloaded from https://www.aminer.cn/oag-2-1. The original data have affiliation data
(`mag_affiliations.zip`), venue data (`mag_venues.zip`), author data (`mag_authors_*.zip`),
paper data (`mag_papers_*.zip`). All data are stored in JSON format.

To construct a graph for GNN training in GraphStorm, we need to preprocess the data
and convert them to the format supported by GraphStorm in
[here](https://github.com/awslabs/graphstorm/wiki/tutorials-own-data#use-own-data).
This data preprocessing is done in the Notebook (`MAG_parser.ipynb`) on a CPU machine.

After the data preprocessing, we also need to run `ComputeBERTEmbed.ipynb` to compute
BERT embeddings on the paper nodes and fos nodes. Please use a GPU machine to generate
BERT embeddings.

Once all the data are processed, run the following command to construct a graph
for distributed GNN training in GraphStorm. The command takes `mag_bert.json`
that specifies the input data for graph construction, constructs the graph,
splits it into 4 partitions and saves the partitions to `mag_bert_constructed3`.
The MAG graph is large, please use
r6a.48xlarge instance to construct the graph. Please prepare a disk to store
some intermediate data in the graph construction process. We can use
`--ext-mem-workspace` to specify where the intermediate data can be stored.

```
python3 -m graphstorm.gconstruct.construct_graph \
			--num-processes 16 \
			--output-dir mag_bert_constructed3 \
			--graph-name mag \
			--num-parts 4 \
			--skip-nonexist-edges \
			--ext-mem-workspace /mnt/raid0/tmp_oag \
			--ext-mem-feat-size 16 \
			--conf-file mag_bert.json
```

After constructing the graph, run the following command for distributed training.
```
python3 -m graphstorm.run.gs_link_prediction \
			--num-trainers 8 --num-servers 4 \
			--part-config mag_bert_constructed/mag.json \
			--ip-config ip_list.txt \
			--cf mag_lp.yaml \
			--num-epochs 1 \
			--save-model-path ~/mag_model \
			--node-feat-name fos:feat paper:feat 
```

## Train GNN model to predict the venue of papers
```
export TOKENIZERS_PARALLELISM=false

python3 -m graphstorm.gconstruct.construct_graph \
			--output-dir mag_min_1parts \
			--graph-name mag \
			--num-parts 1 \
			--skip-nonexist-edges \
			--conf-file mag_min_v0.2.json \
			--add-reverse-edges

python3 -m graphstorm.gconstruct.construct_graph \
			--num-processes 16 \
			--output-dir mag_4parts \
			--graph-name mag \
			--num-parts 4 \
			--skip-nonexist-edges \
			--ext-mem-workspace /mnt/raid0/tmp_oag \
			--ext-mem-feat-size 16 \
			--conf-file mag_v0.2.json \
			--add-reverse-edges
```
### Train GNN model with pre-trained BERT model

```
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_min_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml
```

The accuracy is 37.53%.

### Fine-tune BERT model to predict the venue

```
python3 -m graphstorm.run.gs_node_classification \
		--num-trainers 8 \
		--num-servers 1 \
		--num-samplers 0 \
		--part-config mag_min_4parts/mag.json \
		--ip-config ip_list_4p.txt \
		--cf mag_bert_nc.yaml \
		--save-model-path mag_bert_nc_model
```

The accuracy is 51.64%.

### <a name="bert-ft-gnn"></a>Fine-tune BERT model on the graph data and train GNN model to predict the venue

Fine-tune the BERT model.

```
python3 -m graphstorm.run.gs_link_prediction \
			--num-trainers 8 \
			--num-servers 1 \
			--num-samplers 0 \
			--part-config mag_min_4parts/mag.json \
			--ip-config ip_list_4p.txt \
			--cf mag_bert_ft.yaml \
			--save-model-path mag_bert_lp_model
```

Train the GNN model.

```
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_min_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml \
            --restore-model-path mag_bert_lp_model/epoch-2 \
            --restore-model-layers embed
```

The accuracy is 51.65%.

```
python3 -m graphstorm.run.gs_node_classification \
            --num-trainers 8 \
            --num-servers 4 \
            --num-samplers 0 \
            --part-config mag_min_4parts/mag.json \
            --ip-config ip_list_4p.txt \
            --cf mag_gnn_nc.yaml \
            --batch-size 128 \
            --restore-model-path mag_bert_nc_model/epoch-9/ \
            --restore-model-layers embed
```

The accuracy is 57.41%.

### Co-training BERT and GNN models using GLEM to predict the venue

[GLEM](https://arxiv.org/abs/2210.14709) is a variational EM framework that trains a LM and GNN iteratively for semi-supervised node classification. There are two important pre-requisite for achieve good performance with GLEM

1. The pseudolabeling technique: it predicts pseudolabels on the unlabeled nodes and uses as additional supervision signal for mutual distillation between LM and GNN. This can be enabled by the `--use-pseudolabel true` argument in command line. 
2. Well pre-trained LM and GNN before the co-training: empirically, LM or GNN models that are not well-trained lead to degraded performance when co-training with GLEM directly. Therefore, we suggest user to pre-train the LM and GNN first. This can be achieved by:
	1. Setting `num_pretrain_epochs` in the [yaml config](mag_glem_w_pretrain.yaml). 

	```
	python3 -m graphstorm.run.gs_node_classification \
				--num-trainers 8 \
				--num-servers 4 \
				--num-samplers 0 \
				--part-config mag_min_4parts/mag.json \
				--ip-config ip_list_4p.txt \
				--cf mag_glem_w_pretrain.yaml \
				--use-pseudolabel true
	```

	2. Restoring pretrained model from checkpoints using `--restore-model-path`. In the following example, we restore the GNN trained on fine-tuned BERT model in the [previous section](#bert-ft-gnn). GLEM requires checkpoints of LM and GNN to be in the same path, under separate directories `LM` and `GNN`. It then loads the LM's `node_input_encoder` and GNN's `gnn_encoder` and `decoder`. Since our GNN checkpoint contain both the fine-tuned LM and GNN, we set up softlinks to point both LM and GNN to this checkpiont. 

	```
	# prepare paths to pretrained models:
	mkdir mag_pretrained_models
	ln -s mag_gnn_nc_model/epoch-7 mag_pretrained_models/LM
	ln -s mag_gnn_nc_model/epoch-7 mag_pretrained_models/GNN
	
	# co-training pre-trained LM and GNN with GLEM:
	python3 -m graphstorm.run.gs_node_classification \
				--num-trainers 8 \
				--num-servers 4 \
				--num-samplers 0 \
				--part-config mag_min_4parts/mag.json \
				--ip-config ip_list_4p.txt \
				--cf mag_glem_nc.yaml \
				--use-pseudolabel true \
				--restore-model-path mag_pretrained_models
	```

