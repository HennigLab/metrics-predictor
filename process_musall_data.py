import os, sys

# reqires:
# pip install spikeinterface[full]
# or clone from GitHub and install with pip install -e .

# use local version of spikeinterface
# sys.path.insert(0, os.path.abspath("../spikeinterface/"))

from metrics_predictor.metrics_predictor import MetricsPredictor

# import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
import spikeinterface.qualitymetrics as sqm

import numpy as np
from pathlib import Path

METRIC_LIST_SI = sqm.get_quality_metric_list() + sqm.get_quality_pca_metric_list()
METRIC_LIST = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "amplitude_cutoff",
    "amplitude_median",
    "amplitude_cv",
    "synchrony",
    "firing_range",
    "drift",
    "sd_ratio",
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    #  'nn_isolation', #?  broken!
    #  'nn_noise_overlap', #? broken!
    "silhouette",
]
METRIC_LIST = list(set(METRIC_LIST).intersection(set(METRIC_LIST_SI)))

# sorters to use
run_sorters = [
    "herdingspikes",
    "ironclust",
    "kilosort",
    "kilosort2",
    "kilosort2_5",
    "kilosort3",
    "kilosort4",
    "tridesclous",
    "spykingcircus",
    "mountainsort5",
]

if __name__ == "__main__":
    recording_name = "PV2555_20221020_g0_imec0_short"
    # recording_name = "PV2555_20221020_g0_imec0"
    # recording location
    # recording_folder_name = "/disk/data/musall_lab/Neuropixels/PV2555_20221020/PV2555_20221020_g0/PV2555_20221020_g0_imec0/"
    # root folder to store all outputs
    study_folder_name = "./studies"

    study_folder = Path(study_folder_name)
    # recording_folder = Path(recording_folder_name)
    cached_recording_folder = study_folder / (recording_name + "_preprocessed")
    tmp_sorting_folder = study_folder / (recording_name + "_tmp_sorting")
    output_folder = study_folder / (recording_name + "_sortings")
    if not output_folder.is_dir():
        output_folder.mkdir()

    assert cached_recording_folder.is_dir(), "cached_recording_folder does not exist"
    RX = se.BinaryFolderRecording(cached_recording_folder)
    RX.annotate(is_filtered=True)

    # create a metrics predictor
    MP = MetricsPredictor(recording_name, study_folder=study_folder)

    #

    # read thr original KS2_5 sorting
    # SX = se.KiloSortSortingExtractor(
    #     recording_folder / "spikeinterface_KS2_5_output/sorter_output",
    #     remove_empty_units=False,
    # )
    # print(SX)
    # MP.add_sorting(SX, "kilosort25_musall", RX=RX, recording_name=recording_name)

    for sorter in run_sorters:
        output_folder_sorting = output_folder / sorter
        if os.path.exists(output_folder_sorting / "si_folder.json"):
            print(f"reading sorting in folder {output_folder_sorting}")
            x = ss.NpzSortingExtractor.load(output_folder_sorting)
            print(x)
            MP.add_sorting(x, sorter, RX=RX, recording_name=recording_name)
        else:
            print(f"sorting for {sorter} does not exist")

    # compute all metrics
    MP.compute_metrics(
        recompute=False,
        max_spikes_per_unit=100,
        overwrite_waveforms=False,
        metric_list=METRIC_LIST,
        n_jobs=-1,
        n_pca_components=3,
        verbose=False,
        overwrite_pca=False,
    )

    # compute pairwise agreements at two different match scores
    MP.compute_agreements(match_score=0.5, recompute=False)
    MP.compute_agreements(match_score=0.8, recompute=False)
