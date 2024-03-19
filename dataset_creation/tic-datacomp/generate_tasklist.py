#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
import argparse
import yaml


YEARS = range(2014, 2023)
MONTHS = [
    '201403', '201404', '201407', '201408', '201409', '201410', '201411', '201412',
    '201501', '201502', '201503', '201504', '201505', '201506', '201507', '201508', '201510', '201511',
    '201602', '201604', '201605', '201606', '201607', '201608', '201609', '201610', '201612',
    '201701', '201702', '201703', '201704', '201705', '201706', '201707', '201708', '201709', '201710', '201711', '201712',
    '201801', '201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812',
    '201901', '201902', '201903', '201904', '201905', '201906', '201907', '201908', '201909', '201910', '201911', '201912',
    '202001', '202002', '202003', '202004', '202005', '202006', '202007', '202008', '202009', '202010', '202011', '202012',
    '202101', '202102', '202103', '202104', '202105', '202106', '202107', '202108', '202109', '202110', '202111', '202112',
    '202201', '202205', '202206', '202207', '202208', '202209', '202210', '202211', '202212',
    '202301', '202302', '202303', '202304', '202305', '202306',
]


def get_args():
    """Command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml-path",
        type=str,
        required=True,
        help="Path to the local .yml file to be saved.",
    )
    parser.add_argument(
        "--sample-eval",
        action="store_true",
        help="Generate a restricted subset of tasks for debugging.",
    )
    parser.add_argument(
        "--eval-tasks",
        default="retrieval/yearly,datacompnet/yearly"
        ",retrieval/monthly,datacompnet/monthly",
        help="Generate a restricted subset of tasks.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.sample_eval:
        YEARS = [2014]
        MONTHS = ['201403']

    config = {}

    # Add yearly retrieval evaluations
    if "retrieval/yearly" in args.eval_tasks:
        for year in YEARS:
            config[f"tic/datacomp/retrieval/yearly/{year}"] = {
                "name": f"TiC-DataComp-Retrieval-Yearly-{year}",
                "main_metric": "mean_recall@1",
                "time": 20,
                "tags": ["tic", "retrieval"],
            }

    # Add yearly datacompnet evaluations
    if "datacompnet/yearly" in args.eval_tasks:
        for year in YEARS:
            config[f"tic/datacomp/datacompnet/yearly/{year}"] = {
                "name": f"TiC-DataCompNet-Yearly-{year}",
                "num_classes": 1000,
                "time": 20,
                "tags": ["tic", "classification"],
            }

    # Add monthly retrieval evaluations
    if "retrieval/monthly" in args.eval_tasks:
        for month in MONTHS:
            config[f"tic/datacomp/retrieval/monthly/{month}"] = {
                "name": f"TiC-DataComp-Retrieval-Monthly-{month}",
                "main_metric": "mean_recall@1",
                "time": 20,
                "tags": ["tic", "retrieval"],
            }

    # Add monthly datacompnet evaluations
    if "datacompnet/monthly" in args.eval_tasks:
        for month in MONTHS:
            config[f"tic/datacomp/datacompnet/monthly/{month}"] = {
                "name": f"TiC-DataCompNet-Monthly-{month}",
                "num_classes": 1000,
                "time": 20,
                "tags": ["tic", "classification"],
            }

    with open(args.yaml_path, "w") as f:
        yaml.dump(config, f)
