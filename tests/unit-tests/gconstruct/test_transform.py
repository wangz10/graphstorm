"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import pytest
import inspect

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from scipy.special import erfinv

from graphstorm.gconstruct.transform import (_get_output_dtype,
                                             NumericalMinMaxTransform,
                                             Noop,
                                             RankGaussTransform,
                                             CategoricalTransform,
                                             BucketTransform)
from graphstorm.gconstruct.transform import (_check_label_stats_type,
                                             collect_label_stats,
                                             CustomLabelProcessor,
                                             ClassificationProcessor)
from graphstorm.gconstruct.transform import (LABEL_STATS_FIELD,
                                             LABEL_STATS_FREQUENCY_COUNT)

def test_get_output_dtype():
    assert _get_output_dtype("float16") == np.float16
    assert _get_output_dtype("float32") == np.float32
    assert_raises(Exception, _get_output_dtype, "int32")

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32, np.float16])
def test_fp_min_max_bound(input_dtype):
    feats = np.random.randn(100).astype(input_dtype)
    feats[0] = 10.
    feats[1] = -10.
    transform = NumericalMinMaxTransform("test", "test")
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1

    feats = np.random.randn(100).astype(input_dtype)
    res_dtype = np.float32 if input_dtype == np.cfloat else input_dtype
    fifo = np.finfo(res_dtype)
    feats[0] = fifo.max
    feats[1] = -fifo.max
    transform = NumericalMinMaxTransform("test", "test",
                                         out_dtype=input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], np.finfo(res_dtype).max)
    assert_equal(min_val[0], -np.finfo(res_dtype).max)

    if input_dtype == np.float16:
        feats = np.random.randn(100).astype(input_dtype)
        fifo = np.finfo(np.float32)
        feats[0] = fifo.max
        feats[1] = -fifo.max
        transform = NumericalMinMaxTransform("test", "test",
                                             out_dtype=input_dtype)
        max_val, min_val = transform.pre_process(feats)["test"]
        assert len(max_val.shape) == 1
        assert len(min_val.shape) == 1
        assert max_val[0].dtype == np.float16
        assert min_val[0].dtype == np.float16
        assert_equal(max_val[0], np.finfo(np.float16).max)
        assert_equal(min_val[0], -np.finfo(np.float16).max)

        feats = np.random.randn(100).astype(input_dtype)
        fifo = np.finfo(np.float32)
        feats[0] = fifo.max
        feats[1] = -fifo.max
        transform = NumericalMinMaxTransform("test", "test",
                                             max_bound=fifo.max,
                                             min_bound=-fifo.max,
                                             out_dtype=input_dtype)
        max_val, min_val = transform.pre_process(feats)["test"]
        assert transform._max_bound == np.finfo(np.float16).max
        assert transform._min_bound == -np.finfo(np.float16).max
        assert len(max_val.shape) == 1
        assert len(min_val.shape) == 1
        assert max_val[0].dtype == np.float16
        assert min_val[0].dtype == np.float16
        assert_equal(max_val[0], np.finfo(np.float16).max)
        assert_equal(min_val[0], -np.finfo(np.float16).max)


@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
def test_fp_transform(input_dtype):
    # test NumericalMinMaxTransform pre-process
    transform = NumericalMinMaxTransform("test", "test")
    feats = np.random.randn(100).astype(input_dtype)

    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], max_v)
    assert_equal(min_val[0], min_v)

    feats = np.random.randn(100, 1).astype(input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], max_v)
    assert_equal(min_val[0], min_v)

    feats = np.random.randn(100, 10).astype(input_dtype)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert len(max_val) == 10
    assert len(min_val) == 10
    for i in range(10):
        max_v = np.amax(feats[:,i]).astype(np.float32)
        min_v = np.amin(feats[:,i]).astype(np.float32)
        assert_equal(max_val[i], max_v)
        assert_equal(min_val[i], min_v)

    feats = np.random.randn(100).astype(input_dtype)
    feats[0] = 10.
    feats[1] = -10.
    transform = NumericalMinMaxTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], 5.)
    assert_equal(min_val[0], -5.)

    feats = np.random.randn(100, 1).astype(input_dtype)
    feats[0][0] = 10.
    feats[1][0] = -10.
    transform = NumericalMinMaxTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)["test"]
    max_v = np.amax(feats).astype(np.float32)
    min_v = np.amin(feats).astype(np.float32)
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert_equal(max_val[0], 5.)
    assert_equal(min_val[0], -5.)

    feats = np.random.randn(100, 10).astype(input_dtype)
    feats[0] = 10.
    feats[1] = -10.
    transform = NumericalMinMaxTransform("test", "test", max_bound=5., min_bound=-5.)
    max_val, min_val = transform.pre_process(feats)["test"]
    assert len(max_val.shape) == 1
    assert len(min_val.shape) == 1
    assert len(max_val) == 10
    assert len(min_val) == 10
    for i in range(10):
        max_v = np.amax(feats[:,i])
        min_v = np.amin(feats[:,i])
        assert_equal(max_val[i], 5.)
        assert_equal(min_val[i], -5.)

    # Test collect info
    transform_conf = {
        "name": "max_min_norm"
    }
    transform = NumericalMinMaxTransform("test", "test", transform_conf=transform_conf)
    info = [(np.array([1.]), np.array([-1.])),
            (np.array([2.]), np.array([-0.5])),
            (np.array([0.5]), np.array([-0.1]))]
    transform.update_info(info)
    assert len(transform._max_val) == 1
    assert len(transform._min_val) == 1
    assert_equal(transform._max_val[0], 2.)
    assert_equal(transform._min_val[0], -1.)
    assert 'max_val' in transform_conf
    assert 'min_val' in transform_conf
    assert_equal(np.array(transform_conf['max_val']), 2.)
    assert_equal(np.array(transform_conf['min_val']), -1.)

    info = [(np.array([1., 2., 3.]), np.array([-1., -2., 0.5])),
            (np.array([2., 1., 3.]), np.array([-0.5, -3., 0.1])),
            (np.array([0.5, 3., 1.]), np.array([-0.1, -2., 0.3]))]
    transform.update_info(info)
    assert len(transform._max_val) == 3
    assert len(transform._min_val) == 3
    assert_equal(transform._max_val[0], 2.)
    assert_equal(transform._min_val[0], -1.)
    assert 'max_val' in transform_conf
    assert 'min_val' in transform_conf
    assert_equal(np.array(transform_conf['max_val']),
                 np.array([2.,3.,3.]))
    assert_equal(np.array(transform_conf['min_val']),
                 np.array([-1.,-3.,0.1]))

    transform_conf = {
        "name": "max_min_norm",
        "max_val": [1.,1.,1.],
        "min_val": [-1.,-1.,-1.]
    }
    transform = NumericalMinMaxTransform("test", "test",
                                        max_val=transform_conf['max_val'],
                                        min_val=transform_conf['min_val'],
                                        transform_conf=transform_conf)
    feats = 2 * np.random.randn(10, 3).astype(input_dtype)
    feats[0][0] = 2
    feats[0][1] = -2
    info = transform.pre_process(feats)
    max_val = np.array(transform_conf['max_val'])
    min_val = np.array(transform_conf["min_val"])
    assert_equal(info["test"][0], max_val)
    assert_equal(info["test"][1], min_val)
    transform.update_info([info["test"]])
    assert_equal(np.array(transform_conf['max_val']),
                 np.array([1.,1.,1.]))
    assert_equal(np.array(transform_conf['min_val']),
                 np.array([-1.,-1.,-1.]))
    result = transform(feats)
    true_result = (feats - min_val) / (max_val - min_val)
    true_result[true_result > 1] = 1
    true_result[true_result < 0] = 0
    assert_almost_equal(result["test"].astype(input_dtype), true_result)

    transform_conf = {
        "name": "max_min_norm",
        "min_val": [-1.,-1.,-1.]
    }
    transform = NumericalMinMaxTransform("test", "test",
                                        min_val=transform_conf['min_val'],
                                        transform_conf=transform_conf)
    info = transform.pre_process(feats)
    max_val = info["test"][0]
    min_val = np.array(transform_conf['min_val'])
    assert_equal(info["test"][0], max_val)
    transform.update_info([info["test"]])
    assert_equal(np.array(transform_conf['max_val']),
                 max_val)
    assert_equal(np.array(transform_conf['min_val']),
                 np.array([-1.,-1.,-1.]))
    result = transform(feats)
    true_result = (feats - min_val) / (max_val - min_val)
    true_result[true_result > 1] = 1
    true_result[true_result < 0] = 0
    assert_almost_equal(result["test"].astype(input_dtype), true_result)

    transform_conf = {
        "name": "max_min_norm",
        "max_val": [1.,1.,1.]
    }
    transform = NumericalMinMaxTransform("test", "test",
                                        max_val=transform_conf['max_val'],
                                        transform_conf=transform_conf)
    info = transform.pre_process(feats)
    max_val = np.array(transform_conf['max_val'])
    min_val = info["test"][1]
    assert_equal(info["test"][0], max_val)
    transform.update_info([info["test"]])
    assert_equal(np.array(transform_conf['max_val']),
                 np.array([1.,1.,1.]))
    assert_equal(np.array(transform_conf['min_val']),
                 min_val)
    result = transform(feats)
    true_result = (feats - min_val) / (max_val - min_val)
    true_result[true_result > 1] = 1
    true_result[true_result < 0] = 0
    assert_almost_equal(result["test"].astype(input_dtype), true_result)

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_fp_min_max_transform(input_dtype, out_dtype):
    transform = NumericalMinMaxTransform("test", "test", out_dtype=out_dtype)
    max_val = np.array([2.])
    min_val = np.array([-1.])
    transform._max_val = max_val
    transform._min_val = min_val
    feats = np.random.randn(100).astype(input_dtype)
    norm_feats = transform(feats)["test"]
    if out_dtype is not None:
        assert norm_feats.dtype == np.float16
    else:
        assert norm_feats.dtype != np.float16
    feats[feats > max_val] = max_val
    feats[feats < min_val] = min_val
    feats = (feats-min_val)/(max_val-min_val)
    feats = feats if out_dtype is None else feats.astype(out_dtype)
    assert_almost_equal(norm_feats, feats, decimal=5)

    feats = np.random.randn(100, 1).astype(input_dtype)
    norm_feats = transform(feats)["test"]
    if out_dtype is not None:
        assert norm_feats.dtype == np.float16
    else:
        assert norm_feats.dtype != np.float16
    feats[feats > max_val] = max_val
    feats[feats < min_val] = min_val
    feats = (feats-min_val)/(max_val-min_val)
    feats = feats if out_dtype is None else feats.astype(out_dtype)
    assert_almost_equal(norm_feats, feats, decimal=6)

    transform = NumericalMinMaxTransform("test", "test", out_dtype=out_dtype)
    max_val = np.array([2., 3., 0.])
    min_val = np.array([-1., 1., -0.5])
    transform._max_val = max_val
    transform._min_val = min_val
    feats = np.random.randn(10, 3).astype(input_dtype)
    norm_feats = transform(feats)["test"]
    if out_dtype is not None:
        assert norm_feats.dtype == np.float16
    else:
        assert norm_feats.dtype != np.float16
    for i in range(3):
        new_feats = feats[:,i]
        new_feats[new_feats > max_val[i]] = max_val[i]
        new_feats[new_feats < min_val[i]] = min_val[i]
        new_feats = (new_feats-min_val[i])/(max_val[i]-min_val[i])
        new_feats = new_feats if out_dtype is None else new_feats.astype(out_dtype)
        assert_almost_equal(norm_feats[:,i], new_feats, decimal=6)


def test_categorize_transform():
    # Test a single categorical value.
    transform_conf = {
        "name": "to_categorical"
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    str_ids = [str(i) for i in np.random.randint(0, 10, 1000)]
    str_ids[0] = None
    str_ids[-1] = None # allow None data
    str_ids = str_ids + [str(i) for i in range(10)]
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 10
    for i in range(10):
        assert str(i) in res["test"]

    info = [ np.array([str(i) for i in range(6)]),
            np.array([str(i) for i in range(4, 10)]) ]
    transform.update_info(info)
    feat = np.array([str(i) for i in np.random.randint(0, 10, 100)])
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)
    assert "mapping" in transform_conf
    assert len(transform_conf["mapping"]) == 10
    feat = np.array([None, None]) # transform numpy array with None value.
    cat_feat = transform(feat)
    assert "test" in cat_feat
    assert np.all(cat_feat["test"][0] == 0)
    assert np.all(cat_feat["test"][1] == 0)

    feat = [str(i) for i in np.random.randint(0, 10, 100)]
    feat_with_unknown = feat[:]
    feat_with_unknown[0] = "10"
    feat = np.array(feat)
    feat_with_unknown = np.array(feat_with_unknown)
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for i, (feat, str_i) in enumerate(zip(cat_feat["test"], feat)):
        if i == 0:
            continue
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)

    # Test categorical values with empty strings.
    transform = CategoricalTransform("test1", "test", separator=',')
    str_ids = [f"{i},{i+1}" for i in np.random.randint(0, 9, 1000)] + [",0"]
    str_ids = str_ids + [str(i) for i in range(9)]
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 11

    # Test multiple categorical values.
    transform = CategoricalTransform("test1", "test", separator=',')
    str_ids = [f"{i},{i+1}" for i in np.random.randint(0, 9, 1000)]
    str_ids = str_ids + [str(i) for i in range(9)]
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 10
    for i in range(10):
        assert str(i) in res["test"]

    info = [ np.array([str(i) for i in range(6)]),
            np.array([str(i) for i in range(4, 10)]) ]
    transform.update_info(info)
    feat = np.array([f"{i},{i+1}" for i in np.random.randint(0, 9, 100)])
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_feat in zip(cat_feat["test"], feat):
        # make sure two elements are 1
        i = str_feat.split(",")
        assert feat[int(i[0])] == 1
        assert feat[int(i[1])] == 1
        # after removing the elements, the vector has only 0 values.
        feat[int(i[0])] = 0
        feat[int(i[1])] = 0
        assert np.all(feat == 0)

    # feat contains unknown keys
    feat = [f"{i},{i+1}" for i in np.random.randint(0, 9, 100)]
    feat_with_unknown = feat[:]
    feat_with_unknown[0] = feat[0]+",10"
    feat = np.array(feat)
    feat_with_unknown = np.array(feat_with_unknown)
    cat_feat = transform(feat_with_unknown)
    assert "test" in cat_feat
    for feat, str_feat in zip(cat_feat["test"], feat):
        # make sure two elements are 1
        i = str_feat.split(",")
        assert feat[int(i[0])] == 1
        assert feat[int(i[1])] == 1
        # after removing the elements, the vector has only 0 values.
        feat[int(i[0])] = 0
        feat[int(i[1])] = 0
        assert np.all(feat == 0)

    # Test transformation with existing mapping.
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    str_ids = [str(i) for i in np.random.randint(0, 10, 1000)]
    str_ids = str_ids + [str(i) for i in range(10)]
    res = transform.pre_process(np.array(str_ids))
    assert len(res) == 0

    transform.update_info([])
    feat = np.array([str(i) for i in np.random.randint(0, 10, 100)])
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        idx = transform_conf["mapping"][str_i]
        assert feat[idx] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[idx] = 0
        assert np.all(feat == 0)

    # Test the case when the input ids are mixture of int and strings
    # This may happen when loading csv files.
    transform_conf = {
        "name": "to_categorical"
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    str_ids = [str(i) for i in np.random.randint(0, 10, 1000)]
    str_ids[0] = None
    str_ids[-1] = None # allow None data
    str_ids = str_ids + [i for i in range(10)] # mix strings with ints
    res = transform.pre_process(np.array(str_ids))
    assert "test" in res
    assert len(res["test"]) == 10
    for i in range(10):
        assert str(i) in res["test"]

    info = [np.array([str(i) for i in range(6)]),
            np.array([str(i) for i in range(4, 10)])]
    transform.update_info(info)
    feat = np.array([str(i) for i in np.random.randint(0, 10, 100)])
    feat[0] = int(feat[0])
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)
    assert "mapping" in transform_conf
    assert len(transform_conf["mapping"]) == 10

    # check update_info
    transform_conf = {
        "name": "to_categorical"
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    info = [np.array([i for i in range(6)]),
            np.array([i for i in range(4, 10)])]
    transform.update_info(info)
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)
    assert len(transform_conf["mapping"]) == 10

    # check backward compatible
     # check update_info
    transform_conf = {
        "name": "to_categorical",
        "mapping": {i: i for i in range(10)}
    }
    transform = CategoricalTransform("test1", "test", transform_conf=transform_conf)
    assert len(transform_conf["mapping"]) == 10
    cat_feat = transform(feat)
    assert "test" in cat_feat
    for feat, str_i in zip(cat_feat["test"], feat):
        # make sure one value is 1
        assert feat[int(str_i)] == 1
        # after we set the value to 0, the entire vector has 0 values.
        feat[int(str_i)] = 0
        assert np.all(feat == 0)

@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_noop_transform(out_dtype):
    transform = Noop("test", "test", out_dtype=out_dtype)
    feats = np.random.randn(100).astype(np.float32)
    norm_feats = transform(feats)
    if out_dtype is not None:
        assert norm_feats["test"].dtype == out_dtype
    else:
        assert norm_feats["test"].dtype == np.float32

@pytest.mark.parametrize("input_dtype", [np.cfloat, np.float32])
@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_rank_gauss_transform(input_dtype, out_dtype):
    eps = 1e-6
    transform = RankGaussTransform("test", "test", out_dtype=out_dtype, epsilon=eps)
    feat_0 = np.random.randn(100,2).astype(input_dtype)
    feat_trans_0 = transform(feat_0)['test']
    feat_1 = np.random.randn(100,2).astype(input_dtype)
    feat_trans_1 = transform(feat_1)['test']
    assert feat_trans_0.dtype == np.float32
    assert feat_trans_1.dtype == np.float32
    def rank_gauss(feat):
        lower = -1 + eps
        upper = 1 - eps
        range = upper - lower
        i = np.argsort(feat, axis=0)
        j = np.argsort(i, axis=0)
        j_range = len(j) - 1
        divider = j_range / range
        feat = j / divider
        feat = feat - upper
        return erfinv(feat)

    feat = np.concatenate([feat_0, feat_1])
    feat = rank_gauss(feat)
    new_feat = np.concatenate([feat_trans_0, feat_trans_1])
    trans_feat = transform.after_merge_transform(new_feat)

    if out_dtype is not None:
        assert trans_feat.dtype == np.float16
        assert_almost_equal(feat.astype(np.float16), trans_feat, decimal=3)
    else:
        assert trans_feat.dtype != np.float16
        assert_almost_equal(feat, trans_feat, decimal=4)

def test_custom_label_processor():
    train_idx = np.arange(0, 10)
    val_idx = np.arange(10, 15)
    test_idx = np.arange(15, 20)
    clp = CustomLabelProcessor("test_label", "test", "id", "classification",
                               train_idx, val_idx, test_idx, None)

    split = clp.data_split(np.arange(20))
    assert "train_mask" in split
    assert "val_mask" in split
    assert "test_mask" in split
    assert_equal(np.squeeze(np.nonzero(split["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(split["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(split["test_mask"])), test_idx)

    split = clp.data_split(np.arange(24))
    assert "train_mask" in split
    assert "val_mask" in split
    assert "test_mask" in split
    assert_equal(np.squeeze(np.nonzero(split["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(split["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(split["test_mask"])), test_idx)
    assert len(split["train_mask"]) == 24
    assert len(split["val_mask"]) == 24
    assert len(split["test_mask"]) == 24

    # there is no label
    input_data = {
        "feat": np.random.rand(24),
        "id": np.arange(24),
    }
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), test_idx)

    # there are labels, but not classification
    input_data = {
        "test_label": np.random.randint(0, 5, (24,)),
        "id": np.arange(24),
    }
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), test_idx)
    assert_equal(ret["test"], input_data["test_label"])

    # there labels and _stats_type is frequency count
    input_data = {
        "test_label": np.random.randint(0, 5, (24,)),
        "id": np.arange(24),
    }
    clp = CustomLabelProcessor("test_label", "test", "id", "classification",
                         train_idx, val_idx, test_idx, LABEL_STATS_FREQUENCY_COUNT)
    ret = clp(input_data)
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert_equal(np.squeeze(np.nonzero(ret["train_mask"])), train_idx)
    assert_equal(np.squeeze(np.nonzero(ret["val_mask"])), val_idx)
    assert_equal(np.squeeze(np.nonzero(ret["test_mask"])), test_idx)
    assert_equal(ret["test"], input_data["test_label"])
    stats_info_key = LABEL_STATS_FIELD+"test"
    assert LABEL_STATS_FIELD+"test" in ret
    vals, counts = np.unique(input_data["test_label"][train_idx], return_counts=True)
    assert ret[stats_info_key][0] == LABEL_STATS_FREQUENCY_COUNT
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)

def test_check_label_stats_type():
    stats_type = _check_label_stats_type("regression", LABEL_STATS_FREQUENCY_COUNT)
    assert stats_type is None

    stats_type = _check_label_stats_type("classification", LABEL_STATS_FREQUENCY_COUNT)
    assert stats_type == LABEL_STATS_FREQUENCY_COUNT

    with pytest.raises(Exception):
        stats_type = _check_label_stats_type("classification", "unknown")

def test_collect_label_stats():
    feat_name = LABEL_STATS_FIELD+"test"
    label_stats = [(LABEL_STATS_FREQUENCY_COUNT, np.array([0,1,2,3]), np.array([1,3,5,7]))]
    label_name, stats_type, info = collect_label_stats(feat_name, label_stats)
    assert label_name == "test"
    assert stats_type == LABEL_STATS_FREQUENCY_COUNT
    assert info[0] == 1
    assert info[1] == 3
    assert info[2] == 5
    assert info[3] == 7

    label_stats = [(LABEL_STATS_FREQUENCY_COUNT, np.array([0,2]), np.array([3,4])),
                   (LABEL_STATS_FREQUENCY_COUNT, np.array([0,1,2,3]), np.array([1,3,5,7]))]
    label_name, stats_type, info = collect_label_stats(feat_name, label_stats)
    assert label_name == "test"
    assert stats_type == LABEL_STATS_FREQUENCY_COUNT
    assert info[0] == 4
    assert info[1] == 3
    assert info[2] == 9
    assert info[3] == 7

    with pytest.raises(Exception):
        label_stats = [("unknown", np.array[0,1,2,3], np.array[1,3,5,7])]
        label_name, stats_type, info = collect_label_stats(feat_name, label_stats)

def test_classification_processor():
    clp = ClassificationProcessor("test_label", "test", [0.8,0.1,0.1], LABEL_STATS_FREQUENCY_COUNT)

    # there is no label
    input_data = {
        "test_label": np.random.randint(0, 5, (24,))
    }
    ret = clp(input_data)
    stats_info_key = LABEL_STATS_FIELD+"test"
    assert "test" in ret
    assert "train_mask" in ret
    assert "val_mask" in ret
    assert "test_mask" in ret
    assert stats_info_key in ret
    vals, counts = np.unique(input_data["test_label"][ret["train_mask"].astype(np.bool_)],
                             return_counts=True)
    assert ret[stats_info_key][0] == LABEL_STATS_FREQUENCY_COUNT
    assert_equal(ret[stats_info_key][1], vals)
    assert_equal(ret[stats_info_key][2], counts)


@pytest.mark.parametrize("out_dtype", [None, np.float16])
def test_bucket_transform(out_dtype):
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                 bucket_range=bucket_range, slide_window_size=0, out_dtype=out_dtype)
    feats = np.array([1, 11, 21, 31])
    bucket_feats = transform(feats)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Case for float number
    bucket_range = [1.1, 3.1]
    feats = np.array([0.2, 1.2, 2.2, 3.2])
    transform = BucketTransform("test", "test", 2,
                 bucket_range=bucket_range, slide_window_size=0, out_dtype=out_dtype)
    bucket_feats = transform(feats)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Case with sliding window
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                 bucket_range=bucket_range, slide_window_size=10, out_dtype=out_dtype)
    feats = np.array([1, 11, 21, 31])
    bucket_feats = transform(feats)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Edge case for data on the bucket edge
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                                bucket_range=bucket_range, out_dtype=out_dtype)
    feats = np.array([1, 10, 20, 30])
    bucket_feats = transform(feats)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Edge case for large sliding window
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 3,
                                bucket_range=bucket_range, slide_window_size=20,
                                out_dtype=out_dtype)
    feats = np.array([1, 10, 20, 30])
    bucket_feats = transform(feats)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

    # Edge case for super large sliding window
    bucket_range = [10, 30]
    transform = BucketTransform("test", "test", 2,
                                bucket_range=bucket_range, slide_window_size=100,
                                out_dtype=out_dtype)
    feats = np.array([1, 10, 20, 30])
    bucket_feats = transform(feats)
    if out_dtype is not None:
        assert bucket_feats['test'].dtype == np.float16
    else:
        assert bucket_feats['test'].dtype == np.float32

    feats_tar = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=out_dtype)
    assert_equal(bucket_feats['test'], feats_tar)

if __name__ == '__main__':
    test_categorize_transform()
    test_get_output_dtype()
    test_fp_transform(np.cfloat)
    test_fp_transform(np.float32)
    test_fp_min_max_bound(np.cfloat)
    test_fp_min_max_bound(np.float32)
    test_fp_min_max_bound(np.float16)
    test_fp_min_max_transform(np.cfloat, None)
    test_fp_min_max_transform(np.cfloat, np.float16)
    test_fp_min_max_transform(np.float32, None)
    test_fp_min_max_transform(np.float32, np.float16)
    test_noop_transform(None)
    test_noop_transform(np.float16)
    test_bucket_transform(None)
    test_bucket_transform(np.float16)

    test_rank_gauss_transform(np.cfloat, None)
    test_rank_gauss_transform(np.cfloat, np.float16)
    test_rank_gauss_transform(np.float32, None)
    test_rank_gauss_transform(np.float32, np.float16)

    test_check_label_stats_type()
    test_collect_label_stats()
    test_custom_label_processor()
    test_classification_processor()
