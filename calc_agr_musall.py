# %load_ext autoreload
# %autoreload 2
import os
import numpy as np
import pandas as pd
import glob
import pickle

import datetime # logging

os.environ["KACHERY_CLOUD_DIR"] = "/disk/data/scratch_robyn/kachery"

from pathlib import Path
from metrics_predictor.metrics_predictor import MetricsPredictor

import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spp
from spikeinterface.sorters import read_sorter_folder
from spikeinterface.core import NumpySorting, NpzFolderSorting
from spikeinterface.extractors import read_kilosort
from spikeinterface.comparison.multicomparisons import compare_multiple_sorters
import spikeinterface.widgets as sw
from spikeinterface.comparison import MultiSortingComparison
from spikeinterface.comparison.multicomparisons import compare_multiple_sorters

import utils

from matplotlib import pyplot as plt

# helper
def rmtree(root):
    for p in root.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()
    root.rmdir()



study_folder = './studies/'
sortings_folder = "/disk/data/musall_lab/bonn_data/neuronID/sorters_ouputs"
study_name = 'Musall_lab'
recording_name = '2814_20230912' #'2806_20230914'


MP = MetricsPredictor(study_name, study_folder)

sorters = list(os.walk(sortings_folder+'/'+recording_name+'/studies/'+recording_name+'_sortings'))[0][1]

sortings = []

metrics = []
for sorter in sorters:
    try:
        sorting = se.NumpySorting.load_from_folder(sortings_folder+'/'+recording_name+'/studies/'+recording_name+'_sortings/'+sorter)
    except:
        sorting = se.KiloSortSortingExtractor(sortings_folder+'/'+recording_name+'/studies/'+recording_name+'_sortings/'+sorter)
    print(sorting)
    sortings.append(sorting)

    folder = study_folder+'/'+study_name+f'/sorting_{sorter}_{recording_name}'
    folder
    # check if folder exists
    # if os.path.isdir(folder):
    # # if folder.exists():
    #     rmtree(folder)
    MP.add_sorting(sorting, sorter, recording_name=recording_name)

    # get metrics
    metrics_file = (
        # study_folder / f"metrics_{recording_name}_{sorting_name}.csv"
        # f"/disk/data/musall_lab/bonn_data/neuronID/agreement_scores/{recording_name}_{sorter}.csv"
        f"/disk/data/musall_lab/bonn_data/neuronID/agreement_scores/{recording_name}/metrics_{recording_name}_{sorter}.csv"

    )
    if os.path.exists(metrics_file):
        # print("yes it exists")
        m = pd.read_csv(metrics_file)

        m["sorter"] = sorter #sorting["sorting_name"]
        m["recording"] = recording_name #sorting["recording_name"]
        m["sorter_unit_id"] = sorting.get_unit_ids()
        metrics.append(m)
    
    metrics_save_loc = Path(MP.study_folder / f'sorting_{recording_name}_metrics.csv')
    print(f"saving to {metrics_save_loc}")
    pd.DataFrame(m).to_csv(metrics_save_loc, index=False)
metrics_df = pd.concat(metrics, ignore_index=True)
    
na_dict = utils.print_missing_values(metrics_df, print_result=False)
metrics_df = utils.drop_cols_with_all_nans(metrics_df, na_dict)

###### the bit to get agreement
print("starting calculation for 0.5")
results_folder = MP.study_folder
sortings = [s["sorting"] for s in MP.sortings]
name_list = [s["sorting_name"] for s in MP.sortings]
threshold = 0.5

# trheshold with _ instead of .
threshold_name = f"agreement_{threshold}".replace(".", "_")
# threshold_name

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(Path(results_folder / threshold_name)):
    os.makedirs(Path(results_folder/ threshold_name ))

if not os.path.isfile(Path(results_folder/ threshold_name /'mcmp.pkl')):
    comparison = compare_multiple_sorters(sortings, name_list=name_list, match_score=threshold)
    pickle.dump(comparison, open(Path(results_folder / threshold_name /'mcmp.pkl'), 'wb'))
else:
    comparison = pickle.load(open(Path(results_folder / threshold_name /'mcmp.pkl'), 'rb'))

# again with 0.8
print("starting calculation for 0.8")
threshold = 0.8
# trheshold with _ instead of .
threshold_name = f"agreement_{threshold}".replace(".", "_")
# threshold_name


if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(Path(results_folder / threshold_name)):
    os.makedirs(Path(results_folder/ threshold_name ))

if not os.path.isfile(Path(results_folder/ threshold_name /'mcmp.pkl')):
    comparison = compare_multiple_sorters(sortings, name_list=name_list, match_score=threshold)
    pickle.dump(comparison, open(Path(results_folder / threshold_name /'mcmp.pkl'), 'wb'))
else:
    comparison = pickle.load(open(Path(results_folder / threshold_name /'mcmp.pkl'), 'rb'))