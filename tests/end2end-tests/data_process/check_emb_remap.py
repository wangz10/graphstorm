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
import os
import argparse
import json
import torch as th

from graphstorm.gconstruct.file_io import read_data_parquet
from numpy.testing import assert_equal

def main(args):
    ntype0 = "n0"
    ntype1 = "n1"

    emb_path = args.remap_output

    ntype0_emb_path = os.path.join(emb_path, ntype0)
    data = read_data_parquet(
        os.path.join(ntype0_emb_path, "emb.part00000_00000.parquet"),
        data_fields=["emb", "nid"])

    assert_equal(data["emb"][:,0].astype("str"), data["nid"])
    assert_equal(data["emb"][:,1].astype("str"), data["nid"])
    data = read_data_parquet(
        os.path.join(ntype0_emb_path, "emb.part00001_00000.parquet"),
        data_fields=["emb", "nid"])
    assert_equal(data["emb"][:,0].astype("str"), data["nid"])
    assert_equal(data["emb"][:,1].astype("str"), data["nid"])

    ntype1_emb_path = os.path.join(emb_path, ntype1)
    data = read_data_parquet(
        os.path.join(ntype1_emb_path, "emb.part00000_00000.parquet"),
        data_fields=["emb", "nid"])
    assert_equal(data["emb"][:,0].astype("str"), data["nid"])
    assert_equal(data["emb"][:,1].astype("str"), data["nid"])
    data = read_data_parquet(
        os.path.join(ntype1_emb_path, "emb.part00001_00000.parquet"),
        data_fields=["emb", "nid"])
    assert_equal(data["emb"][:,0].astype("str"), data["nid"])
    assert_equal(data["emb"][:,1].astype("str"), data["nid"])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Check edge prediction remapping")
    argparser.add_argument("--remap-output", type=str, required=True,
                           help="Path to save the generated data")

    args = argparser.parse_args()

    main(args)