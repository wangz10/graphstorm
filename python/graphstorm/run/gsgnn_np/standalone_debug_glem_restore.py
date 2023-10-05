import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnNodePredictionInfer
from graphstorm.eval import GSgnnAccEvaluator, GSgnnRegressionEvaluator
from graphstorm.dataloading import GSgnnNodeInferData, GSgnnNodeDataLoader
from graphstorm.utils import setup_device

def get_evaluator(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "node_regression":
        return GSgnnRegressionEvaluator(config.eval_frequency,
                                        config.eval_metric)
    elif config.task_type == 'node_classification':
        return GSgnnAccEvaluator(config.eval_frequency,
                                 config.eval_metric,
                                 config.multilabel)
    else:
        raise AttributeError(config.task_type + ' is not supported.')

import os
# init the GS dist cluster
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '4321'
gs.initialize(ip_config='127.0.0.1', backend='gloo')

import yaml
from argparse import Namespace


cmd_args = {
    'num_trainers': 1, 
    'num_servers': 1,
    'num_samplers': 0,
    'part_config': '/data/mag_min_1part3/mag.json',
    'ip_config': 'ip_list.txt',
    # 'yaml_config_file': '/develop/graphstorm/examples/mag/mag_glem_from_checkpoints.yaml',
    # 'yaml_config_file': '/develop/graphstorm/examples/mag/mag_glem_nc.yaml',
    'yaml_config_file': '/data/mag_min_4parts/mag_glem_from_checkpoints_4p_nopt_gnn_p_grouping1.yaml',
    'restore_model_path': '/data/mag_min_1part3/mag_bert_nc_model/',
    'local_rank': 0
}
cmd_args = Namespace(**cmd_args)
config = GSConfig(cmd_args)

infer_data = GSgnnNodeInferData(config.graph_name,
                                config.part_config,
                                eval_ntypes=config.target_ntype,
                                node_feat_field=config.node_feat_name,
                                label_field=config.label_field)

model = gs.create_builtin_node_gnn_model(infer_data.g, config, train_task=False)
print(model.training_sparse_embed)
# check trainable params from lm and gnn
part_to_train = 'gnn'
model.toggle('gnn') # train gnn, freeze LM

model.init_optimizer(lr=config.lr, sparse_optimizer_lr=config.sparse_optimizer_lr,
                             weight_decay=config.wd_l2norm,
                             lm_lr=config.lm_tune_lr)

# _sparse_embeds

## debug sparse optimizer
for i, batch in enumerate(dataloader):
    break

g = infer_data.g
def _prepare_batch(input_nodes, seeds, blocks, is_labeled=True):
    if not isinstance(input_nodes, dict):
        assert len(g.ntypes) == 1
        input_nodes = {g.ntypes[0]: input_nodes}

    input_feats = infer_data.get_node_feats(input_nodes, device)
    lbl = None
    if is_labeled:
        lbl = infer_data.get_labels(seeds, device)
    blocks = [block.to(device) for block in blocks]
    return input_nodes, input_feats, blocks, lbl

input_nodes, seeds, blocks = batch
input_nodes, input_feats, blocks, lbl = _prepare_batch(*batch[0])

sparse_embeds = model.lm.get_sparse_params()
# type(sparse_embeds[0].weight) : dgl.distributed.dist_tensor.DistTensor
# sum(p._tensor.numel() for p in model.lm.get_sparse_params() if p._tensor.requires_grad)

# only these are trainable when model.toggle('gnn'):
sum(p.numel() for p in model.gnn.get_dense_params() if p.requires_grad)
sum(p.numel() for p in model.gnn.gnn_encoder.parameters() if p.requires_grad)
sum(p.numel() for p in model.gnn.decoder.parameters() if p.requires_grad)


model.toggle('lm') # train lm, freeze gnn
# these are trainable:
sum(p.numel() for p in model.lm.node_input_encoder.proj_matrix.parameters() if p.requires_grad)
sum(p.numel() for p in model.lm.node_input_encoder.input_projs.parameters() if p.requires_grad)
sum(p.numel() for p in model.lm.node_input_encoder._lm_models.parameters() if p.requires_grad)
sum(p.numel() for p in model.lm.decoder.parameters() if p.requires_grad)

model.lm.freeze_input_encoder(infer_data[:10])

# sum(p.numel() for p in model.lm.get_lm_params() if p.requires_grad)
# sum(p.numel() for p in model.lm.get_dense_params() if p.requires_grad)
# sum(p.numel() for p in model.gnn.get_lm_params() if p.requires_grad)


# check the dims of LM/GNN in the glem model
# model.lm.decoder.in_dims, model.lm.decoder.out_dims # (128, 1523)
# model.gnn.decoder.in_dims, model.gnn.decoder.out_dims # (128, 1523)

# model.lm.node_input_encoder: GSLMNodeEncoderInputLayer, sparse embedding belongs here
model.lm.node_input_encoder.in_dims, model.lm.node_input_encoder.out_dims # None, 128
# model.lm.node_input_encoder.proj_matrix['venue'].shape: 128, 128
# model.lm.node_input_encoder.input_projs['paper'].shape: 768, 128
# model.lm.node_input_encoder._sparse_embeds['author'].embedding_dim: 128
# model.lm.node_input_encoder._sparse_embeds['author'].num_embeddings: 243477150

# model.gnn.gnn_encoder: graphstorm.model.rgcn_encoder.RelationalGCNEncoder
import torch as th
lm_ckpt = th.load(os.path.join(config.restore_model_path, 'LM/model.bin'))

# lm_ckpt['embed'].keys(): all start from _lm_models._lm_models.s['paper']
# missing proj_matrix, input_projs, _sparse_embeds

gnn_ckpt = th.load(os.path.join(config.restore_model_path, 'GNN/model.bin'))
# gnn_ckpt['embed'].keys(): contains proj_matrix, input_projs, _lm_models._lm_models.['paper'], _lm_models._lm_models.['fox']
# gnn_ckpt['decoder']['decoder'].shape: 128, 1523
# model.restore_model(config.restore_model_path)
model.lm.restore_model(os.path.join(config.restore_model_path, 'GNN'), ['embed'])
model.gnn.restore_model(os.path.join(config.restore_model_path, 'GNN'), ['gnn', 'decoder'])


infer = GSgnnNodePredictionInfer(model, gs.get_rank())
device = setup_device(config.local_rank)

infer.setup_device(device=device)
if not config.no_validation:
    evaluator = get_evaluator(config)
    infer.setup_evaluator(evaluator)
    assert len(infer_data.test_idxs) > 0, \
        "There is not test data for evaluation. " \
        "You can use --no-validation true to avoid do testing"
    target_idxs = infer_data.test_idxs
else:
    assert len(infer_data.infer_idxs) > 0, \
        f"To do inference on {config.target_ntype} without doing evaluation, " \
        "you should not define test_mask as its node feature. " \
        "GraphStorm will do inference on the whole node set. "
    target_idxs = infer_data.infer_idxs
tracker = gs.create_builtin_task_tracker(config, infer.rank)
infer.setup_task_tracker(tracker)
fanout = config.eval_fanout if config.use_mini_batch_infer else []
dataloader = GSgnnNodeDataLoader(infer_data, target_idxs, fanout=fanout,
                                    batch_size=config.eval_batch_size, device=device,
                                    train_task=False,
                                    construct_feat_ntype=config.construct_feat_ntype,
                                    construct_feat_fanout=config.construct_feat_fanout)
# Preparing input layer for training or inference.
# The input layer can pre-compute node features in the preparing step if needed.
# For example pre-compute all BERT embeddings
model.prepare_input_encoder(infer_data)
infer.infer(dataloader, save_embed_path=config.save_embed_path,
            save_prediction_path=config.save_prediction_path,
            use_mini_batch_infer=config.use_mini_batch_infer,
            node_id_mapping_file=config.node_id_mapping_file,
            return_proba=config.return_proba)
