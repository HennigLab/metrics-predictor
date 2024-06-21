import os, sys

# reqires:
# pip install spikeinterface[full]
# pip install tridesclous
# pip install herdingspikes
# pip install mountainsort5

# use local version of spikeinterface
# sys.path.insert(0, os.path.abspath("../spikeinterface/src"))

# specify paths to sorters:
# os.environ["IRONCLUST_PATH"] = "/disk/scratch/mhennig/spikeinterface/ironclust"
# os.environ["KILOSORT_PATH"] = "/disk/scratch/mhennig/spikeinterface/KiloSort/"
# os.environ["KILOSORT2_PATH"] = "/disk/scratch/mhennig/spikeinterface/Kilosort-2-latest/"
# os.environ["NPY_MATLAB_PATH"] = "/disk/scratch/mhennig/spikeinterface/npy-matlab/"
# os.environ["ML_TEMPORARY_DIRECTORY"] = "/disk/scratch/mhennig/tmp/"
# os.environ["TMPDIR"] = "/disk/scratch/mhennig/tmp"
# os.environ["TEMPDIR"] = "/disk/scratch/mhennig/tmp"

# import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.extractors as se
from spikeinterface.preprocessing import bandpass_filter, highpass_filter
from spikeinterface.preprocessing import common_reference, phase_shift
from spikeinterface.sorters import run_sorter
import numpy as np
from pathlib import Path


def get_recording(recording_dir, cached_recording_folder, short_test=False):
    if not cached_recording_folder.is_dir():
        stream_names, _ = se.get_neo_streams("spikeglx", recording_dir)
        RX = se.read_spikeglx(
            recording_dir, stream_name=stream_names[0], load_sync_channel=False
        )
        if (recording_dir / "bad_channel_ids.npy").is_file():
            bad_channels = np.load(recording_dir / "bad_channel_ids.npy")
            bad_channel_ids = [int(b[11:]) for b in bad_channels]
            RX = RX.remove_channels(bad_channel_ids)
            print(f"bad channels read from bad_channel_ids.npy:\n{bad_channel_ids}")
        else:
            bad_channel_ids, channel_labels = si.detect_bad_channels(RX)
            print(f"bad channels detected by SpikeInterface:\n{bad_channel_ids}")
        RX = RX.remove_channels(bad_channel_ids)
        # small recording for testing
        if short_test:
            t2 = 10  # 10 seconds of data only
            RX = RX.frame_slice(0, t2 * RX.get_sampling_frequency())
        # skip these for now as this is already dfone on Musall data:
        RX = phase_shift(RX)
        RX = common_reference(RX, operator="median", reference="global")
        RX = highpass_filter(RX, freq_min=400.0)
        # most sorters require/benefit from bandpass filtered data:
        # RX = bandpass_filter(RX, freq_min=300.0, freq_max=6000.0)
        print(f"saving preprocessed recording: {cached_recording_folder}")
        job_kwargs = dict(n_jobs=4, chunk_duration="1s", progress_bar=True)
        RX = RX.save(folder=cached_recording_folder, format="binary", **job_kwargs)
    else:
        print(f"using cached filtered recording in folder {cached_recording_folder}")
        RX = se.BinaryFolderRecording(cached_recording_folder)
        RX.annotate(is_filtered=True)
    return RX


if __name__ == "__main__":
    # use small data segment for testing if True
    testing = True
    
    # identifier of recording
    recording_name = "PV2555_20221020_g0_imec0_short" # "PV2555_20221020_g0_imec0"
    # recording location
    recording_folder_name = "/disk/data/musall_lab/Neuropixels/PV2555_20221020/PV2555_20221020_g0/PV2555_20221020_g0_imec0/"
    # root folder to store all outputs
    study_folder_name = "studies"

    # sorters to run
    run_sorters = [
        "herdingspikes",
        "ironclust",
        "kilosort",
        "kilosort2",
        # "kilosort2_5",
        "kilosort3",
        "kilosort4", # previously commented
        # "pykilosort",
        "tridesclous",
        # "tridesclous2",
        # "spykingcircus",
        # "spykingcircus2",
        "mountainsort5",
    ]

    # setup paths
    study_folder = Path(study_folder_name)
    if not study_folder.is_dir():
        study_folder.mkdir()
    recording_folder = Path(recording_folder_name)
    cached_recording_folder = study_folder / (recording_name + "_preprocessed")
    tmp_sorting_folder = study_folder / (recording_name + "_tmp_sorting")
    output_folder = study_folder / (recording_name + "_sortings")
    if not output_folder.is_dir():
        output_folder.mkdir()

    # load recording
    RX = get_recording(recording_folder, cached_recording_folder, testing)

    # identify local installs of sorters
    run_sorters = list(set(run_sorters) & set(ss.available_sorters()))
    run_sorters_installed = list(set(run_sorters) & set(ss.installed_sorters()))
    run_sorters_not_installed = list(set(run_sorters) - set(run_sorters_installed))
    print(f"sorters to run in a container:\n{run_sorters_not_installed}")
    print(f"sorters to run from local install: \n{run_sorters_installed}")

    # run sorters in containers
    for sorter in run_sorters_not_installed:
        tmp_folder_sorting = tmp_sorting_folder / sorter
        output_folder_sorting = output_folder / sorter
        if not os.path.exists(output_folder_sorting):
            print(f"running {sorter} in container")
            s = run_sorter(
                sorter,
                RX,
                output_folder=tmp_folder_sorting,
                singularity_image=True,  # set to docker_image=True if needed
                verbose=True,
                delete_container_files=False,
                remove_existing_folder=False,
                delete_output_folder=False,
                raise_error=False,
            )
            print(s)
            if s:
                s.save(folder=str(output_folder_sorting))
        else:
            print(f"not running {sorter} because {output_folder_sorting} exists")

    # run installed sorters
    for sorter in run_sorters_installed:
        tmp_folder_sorting = tmp_sorting_folder / sorter
        output_folder_sorting = output_folder / sorter
        if not os.path.exists(output_folder_sorting):
            print(f"running {sorter} locally")
            s = run_sorter(
                sorter,
                RX,
                output_folder=tmp_folder_sorting,
                verbose=True,
                remove_existing_folder=True,
                delete_output_folder=False,
                raise_error=False,
                # shared_memory=False,
                # job_kwargs=dict(n_jobs=1, chunk_duration="1s", progress_bar=False),
            )
            print(s)
            if s:
                s.save(folder=str(output_folder_sorting))
        else:
            print(f"not running {sorter} because {output_folder_sorting} exists")
