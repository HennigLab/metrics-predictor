import os, sys

# reqires:
# pip install spikeinterface[full]
# or clone from GitHub and install with pip install -e .

# use local version of spikeinterface
# sys.path.insert(0, os.path.abspath("../spikeinterface/"))

from metrics_predictor.metrics_predictor import MetricsPredictor

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
import spikeinterface.qualitymetrics as sqm
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost

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

def get_template_metrics(MP):
    """
    Compute template metrics for a given sorting output.

    Parameters
    ----------
    recording_name : str
        Name of the recording.
    filtered_rec : RecordingExtractor
        The filtered recording extractor.
    sorting : SortingExtractor
        The sorting extractor.
    sorting_name : str
        Name of the sorting output.
    waveform_cache_folder : str
        Folder to save the waveform data. Expects pre-computed waveforms will be saved in this folder (sub-directory "processed"). Saves 
    
    Returns
    -------
    None
    """

    for sorting_item in MP.sortings:

        recording_name = sorting_item["recording_name"]
        filtered_rec = sorting_item["recording"]
        sorting_name = sorting_item["sorting_name"]
        sorting = sorting_item["sorting"]
        
        # remove empty units
        sorting_rm = sorting.remove_empty_units()
        # print(sorting_rm)
        
        # Create or load waveform extraction
        # waveform_cache_folder = "." + "/waveforms_" + recording_name + "_" + sorting_name # replacing catgt output
        waveform_cache_folder = MP.cache_folder + "/waveforms_" + recording_name + "_" + sorting_name # save location once MP.compute_metrics is run
        
        # save_waveform_folder = os.path.join(waveform_cache_folder, 'processed')
        if Path(waveform_cache_folder).exists():
            we = si.load_waveforms(waveform_cache_folder)
        else:
            # waverforms are not in expected location...
            we = si.extract_waveforms(filtered_rec, sorting_rm, folder=waveform_cache_folder, overwrite=True)
        
        # Postprocess metrics
        if "spike_amplitudes" not in we.get_available_extension_names():
            print("\tComputing spike amplitides")
            _ = spost.compute_spike_amplitudes(we)
        else:
            print("\tSpike amplitudes already computed")
        
        # Compute spike locations needed for drift metrics
        if "spike_locations" not in we.get_available_extension_names():
            print("\tComputing spike locations")
            _ = spost.compute_spike_locations(we, method="monopolar_triangulation")
        else:
            print("\tSpike locations already computed")

        # Compute principal components needed for PCA metrics
        if "principal_components" not in we.get_available_extension_names():
            print("\tComputing PCA")
            _ = spost.compute_principal_components(we)
        else:
            print("\tPCA already computed")

        # Compute unit locations
        if "unit_locations" not in we.get_available_extension_names():
            print("\tComputing unit locations")
            _ = spost.compute_unit_locations(we)
        else:
            print("\tUnit locations already computed")

        # Compute correlograms
        if "correlograms" not in we.get_available_extension_names():
            print("\tComputing correlograms")
            _ = spost.compute_correlograms(we)
        else:
            print("\tCorrelograms already computed")

        # Compute template similarity
        if "template_similarity" not in we.get_available_extension_names():
            print("\tComputing template similarity")
            _ = spost.compute_template_similarity(we)
        else:
            print("\tTemplate similarity already computed")
        
        # metrics_folder = os.path.join(save_waveform_folder, 'metrics')
        # put in same location as metrics
        metrics_folder = (
            MP.study_folder # / f"metrics_{MP.recording_name}_{sorting_name}.csv"
        )

        # Check if the folder exists - not necessary with this structure
        # if os.path.exists(metrics_folder):
            # Check if the folder is empty
            # if not os.listdir(metrics_folder):
                # Remove the empty folder
                # os.rmdir(metrics_folder)
            # else:
            #     # If the folder is not empty, exit or handle the case as needed
            #     print(f"Folder '{metrics_folder}' is not empty. Exiting...")
            #     exit()  # or handle the case appropriately
        # Recreate the metrics folder
        # os.makedirs(metrics_folder)

        print("\tComputing template metrics")
        tm = spost.compute_template_metrics(we, include_multi_channel_metrics=True)
        tm.to_csv(os.path.join(metrics_folder, "template_metrics.csv"))

    return 


if __name__ == "__main__":

    recording_name = "PV2555_20221020_g0_imec0" # insert as appropriate
    recording_folder_name = "/disk/data/musall_lab/Neuropixels/PV2555_20221020/PV2555_20221020_g0/PV2555_20221020_g0_imec0/" # recording location
    
    study_folder_name = "./studies" # root folder to store all outputs



    # set up paths
    study_folder = Path(study_folder_name)
    recording_folder = Path(recording_folder_name)
    cached_recording_folder = study_folder / (recording_name + "_preprocessed")
    tmp_sorting_folder = study_folder / (recording_name + "_tmp_sorting")
    output_folder = study_folder / (recording_name + "_sortings")
    if not output_folder.is_dir():
        output_folder.mkdir()

    # load recording as per cache
    assert cached_recording_folder.is_dir(), "cached_recording_folder does not exist"
    RX = se.BinaryFolderRecording(cached_recording_folder) # this should be the filtered recording
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
    MP.cache_folder = "." # output including waveforms will go here, so check!
    # folder = MP.cache_folder + "/waveforms_" + recording_name + "_" + sorting_name # this is where compute metrics will put waveforms
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
    
    get_template_metrics(MP)



    # compute pairwise agreements at two different match scores
    MP.compute_agreements(match_score=0.5, recompute=False)
    MP.compute_agreements(match_score=0.8, recompute=False)
