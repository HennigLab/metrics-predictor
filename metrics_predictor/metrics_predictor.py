import os

from pathlib import Path
import pickle

import spikeinterface.full as si

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
)

from .utils import (
    get_standard_col_name,
    add_col_for_agr_subset,
    print_missing_values,
    drop_cols_with_all_nans,
)

sorter_shorthand = {
    "mountainsort": "MS",
    "MountainSort4": "MS4",
    "mountainsort4": "MS4",
    "mountainsort5": "MS5",
    "tridesclous": "TC",
    "ironclust": "IC",
    "Ironclust": "IC",
    "IronClust": "IC",
    "hdsort": "HDS",
    "HDsort": "HDS",
    "spykingcircus": "SC",
    "herdingspikes": "HS",
    "HerdingSpikes2": "HS",
    "IronClust": "IC",
    "JRClust": "JR",
    "MountainSort4": "MS4",
    "SpykingCircus": "SC",
    "Tridesclous": "TC",
    "KiloSort": "KS",
    "kilosort2": "KS2",
    "KiloSort2": "KS2",
    "Kilosort2": "KS2",
    "kilosort25": "KS25",
    "KiloSort25": "KS25",
    "Kilosort25": "KS25",
}

METRIC_LIST = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violations_ratio",
    "isi_violations_count",
    "rp_contamination",
    "rp_violations",
    "sliding_rp_violation",
    "amplitude_cutoff",
    "amplitude_median",
    "amplitude_cv",
    "sync_spike_2",
    "sync_spike_4",
    "sync_spike_8",
    "firing_range",
    "drift_ptp",
    "drift_std",
    "drift_mad",
    "isolation_distance",
    "l_ratio",
    "d_prime",
    "nearest_neighbor",
    "nn_isolation",
    "nn_noise_overlap",
    "silhouette",
    "nn_hit_rate",
    "nn_miss_rate",
    "nn_unit_id",
    "peak_to_valley",
    "peak_trough_ratio",
    "half_width",
    "repolarization_slope",
    "recovery_slope",
    "num_positive_peaks",
    "num_negative_peaks",
    "velocity_above",
    "velocity_below",
    "exp_decay",
    "spread",
]

skew_to_log = [
    "num_spikes",
    "firing_rate",
    # "isolation_distance",
    # 'snr',
    # "isi_violations_ratio",
    # "isi_violations_count",
    # "amplitude_median",
    # "rp_violations",
    # "amplitude_cutoff",
]

UNIT_INFO = ["sorter", "sorter_unit_id", "recording"]
AGREEMENT_PREFIX = "agreement"
GROUND_TRUTH_PREFIX = "gt_comp_"


class MetricsPredictor:
    """
    Predict unit quality from quality metrics

    Parameters
    ----------
    study_name: str
        name of study, if None use current date and time
    study_folder: str or Path
        folder where study is stored, defaults to ./studies if None
    verbose: bool
        if True, print more

    """

    def __init__(
        self, study_name=None, study_folder=None, cache_folder=None, verbose=True
    ) -> None:
        self.model = None
        self.sortings = None
        self.metrics_df = None
        self.waveform_extractors = None
        self.train_df = None
        self.test_df = None
        self.verbose = verbose
        
        self.skew_to_log = skew_to_log

        if study_name is None:
            # use date and time to create a study name if none is provided
            from datetime import datetime

            now = datetime.now()
            study_name = now.strftime("%Y_%m_%d_%H_%M_%S")
        self.study_name = study_name

        if cache_folder is None:
            self.cache_folder = Path(study_folder) / (study_name + "_cache")
        else:
            self.cache_folder = cache_folder

        if study_folder is None:
            self.study_folder = Path("./studies/" + study_name)
        else:
            self.study_folder = Path(study_folder) / study_name

        for p in self.study_folder.parents:
            if not os.path.isdir(p):
                os.mkdir(p)
        if not os.path.isdir(self.study_folder):
            os.mkdir(self.study_folder)

    def print_study_info(self) -> None:
        """
        Print information about the study
        """
        print(f"study name: {self.study_name}")
        print(f"study folder: {self.study_folder}")
        if self.sortings is not None:
            print(f"number of sorters: {len(self.sortings)}")
        if self.metrics_df is not None:
            self._separate_cols_to_categories(verbose=True)
            print(f"number of units: {len(self.metrics_df)}")
            print("Class imbalance of targets:\n----------------")
            columns = []
            for col in self.metrics_df.columns:
                if col.startswith("agreement"):
                    columns.append(col)
                if col.startswith("gt_comp"):
                    columns.append(col)
            if columns:
                print(
                    self.metrics_df.groupby("recording")[columns]
                    .agg(["mean"])
                    .to_string()
                )

    def get_study_info(self):
        """
        Get information about the study

        Returns
        -------
        metrics: list
            list of metrics
        unit_info: list
            list of unit info columns
        agreement_targets: dict
            dictionary of agreement targets
        gt_targets: list
            list of ground truth targets
        """
        if self.metrics_df is not None:
            return self._separate_cols_to_categories()
        else:
            return [], [], {}, []

    def add_sorting(self, SX, sorting_name, RX=None, recording_name=None):
        """
        Add a sorting to the study. Sortings have to be provided as SpikeInterface sorting extractors.

        Parameters
        ----------
        SX: SortingExtractor
            sorting extractor
        sorting_name: str
            name of sorter or other identifier
        RX: RecordingExtractor
            corresponding recording extractor, required to compute metrics, can be omitted when metrics are already cached
        """
        if self.sortings is None:
            self.sortings = []
        sorting = {}
        sorting["sorting_name"] = sorting_name
        sorting["sorting"] = SX
        sorting["recording"] = RX
        sorting["recording_name"] = recording_name
        self.sortings.append(sorting)

        if self.metrics_df is None:
            self.metrics_df = pd.DataFrame()

        m = {}
        m["sorter"] = sorting["sorting_name"]
        m["recording"] = sorting["recording_name"]
        m["sorter_unit_id"] = sorting["sorting"].get_unit_ids()
        print(f'adding {len(m["sorter_unit_id"])} units from {m["sorter"]} to study')
        # append m to metrics_df
        self.metrics_df = pd.concat(
            [self.metrics_df, pd.DataFrame(m)], ignore_index=True
        )

    def get_sorting_names(self, recording_name=None):
        assert self.sortings is not None, "no sortings added to study"
        if not recording_name:
            return [s["sorting_name"] for s in self.sortings]
        else:
            return [
                s["sorting_name"]
                for s in self.sortings
                if s["recording_name"] == recording_name
            ]

    def compute_metrics(
        self,
        n_jobs=-1,
        recompute=False,
        overwrite_sorting_analyzer=False,
        max_spikes_per_unit=100,
        n_pca_components=2,
        metric_list=None,
        # template_metric_list=None,
        remove_outliers=True,
        verbose=False,
    ) -> None:
        """
        Compute quality metrics for all sortings in study, or load cached metrics.
        """
        assert self.sortings is not None, "no sortings added to study"

        if metric_list is None:
            metric_list = (
                si.get_quality_metric_list() + si.get_quality_pca_metric_list()
            )

        for sorting in self.sortings:
            recording_name = sorting["recording_name"]
            sorting_name = sorting["sorting_name"]
            metrics_file = (
                self.study_folder / f"metrics_{recording_name}_{sorting_name}.csv"
            )
            if os.path.exists(metrics_file) and not recompute:
                if verbose:
                    print(
                        f'loading cached metrics for {sorting["recording_name"]}/{sorting["sorting_name"]} from {metrics_file}'
                    )
                m = pd.read_csv(metrics_file)
                mask = (self.metrics_df["recording"] == sorting["recording_name"]) & (
                    self.metrics_df["sorter"] == sorting["sorting_name"]
                )
                for k in m.keys():
                    if k not in self.metrics_df.keys():
                        self.metrics_df[k] = None
                    self.metrics_df.loc[mask, k] = list(m[k])
            else:
                if verbose:
                    print(
                        f'creating sorting analyzer for {sorting["recording_name"]}/{sorting["sorting_name"]}'
                    )
                self.sorting_analyzer = self._create_sorting_analyzer(
                    sorting,
                    max_spikes_per_unit=max_spikes_per_unit,
                    n_jobs=1,
                    overwrite=overwrite_sorting_analyzer,
                    cache_folder=self.cache_folder,  # type: ignore
                    n_pca_components=n_pca_components,
                    verbose=verbose,
                )
                if verbose:
                    print(
                        f'computing quality metrics for {sorting["recording_name"]}/{sorting["sorting_name"]}'
                    )
                m = self.sorting_analyzer.compute(
                    "quality_metrics",
                    metric_names=metric_list,
                    n_jobs=n_jobs,
                    verbose=verbose,
                ).get_data()
                if verbose:
                    print(
                        f'computing template metrics for {sorting["recording_name"]}/{sorting["sorting_name"]}'
                    )
                mt = self.sorting_analyzer.compute(
                    "template_metrics",
                    n_jobs=n_jobs,
                    verbose=verbose,
                ).get_data()
                m_all = m.join(mt)
                mask = (self.metrics_df["recording"] == sorting["recording_name"]) & (
                    self.metrics_df["sorter"] == sorting["sorting_name"]
                )
                for k in m_all.keys():
                    if k not in self.metrics_df.keys():
                        self.metrics_df[k] = 0.0
                    self.metrics_df.loc[mask, k] = m_all[k].values.astype(float)
                if verbose:
                    print(f"saving metrics to {metrics_file}")
                pd.DataFrame(m_all).to_csv(metrics_file, index=False)
        na_dict = print_missing_values(self.metrics_df, print_result=False)
        self.metrics_df = drop_cols_with_all_nans(
            self.metrics_df, na_dict, print_result=True
        )
        if remove_outliers:
            if verbose:
                print("removing outliers")
            metrics, _, _, _ = self.get_study_info()
            for i, m in enumerate(metrics):
                data = self.metrics_df[m]
                p = np.nanpercentile(data, (0.01 * 100, 100 * (1 - 0.01)))
                inds = (data < 2 * p[0]) | (data > p[1] * 2)
                if verbose and np.sum(inds) > 0:
                    n = np.sum(inds)
                    print(f"{m} has {n} outliers")
                self.metrics_df.loc[np.where(inds)[0], m] = np.nan

    def get_metrics(self) -> pd.DataFrame:
        """
        Get metrics dataframe
        """
        assert self.metrics_df is not None, "compute metrics before getting them"
        return self.metrics_df

    def compute_agreements(
        self, match_score=0.5, delta_time=0.4, chance_score=0.1, recompute=False
    ) -> None:
        """
        Compute agreements for all sortings in study, or load cached agreements.

        Parameters
        ----------
        match_score: float
            score to use for agreement
        recompute: bool
            if True, force recompute agreements
        """
        assert self.metrics_df is not None, "compute metrics before adding agreements"
        recordings = set([sorting["recording_name"] for sorting in self.sortings])
        for rec in recordings:
            print(f"Agreements for recording {rec}")
            sorting_names = self.get_sorting_names(recording_name=rec)
            col_name = get_standard_col_name(
                sorting_names, sorter_shorthand, match_score
            )
            if len(self.get_sorting_names(recording_name=rec)) == 1:
                print(f"Only one sorting for recording {rec}, agreement not computed")
            else:
                agreement_save_folder = self.study_folder / f"{col_name}_{rec}"
                if os.path.exists(agreement_save_folder) and not recompute:
                    print(f"loading cached agreements from {agreement_save_folder}")
                    matching_obj = si.MultiSortingComparison.load_from_folder(
                        agreement_save_folder
                    )
                else:
                    spike_sorter_names = list(sorting_names)
                    spike_sorting_extractor_list = [
                        s["sorting"]
                        for s in self.sortings
                        if s["recording_name"] == rec
                    ]
                    matching_obj = si.compare_multiple_sorters(
                        spike_sorting_extractor_list,
                        name_list=spike_sorter_names,
                        delta_time=delta_time,
                        match_score=match_score,
                        chance_score=chance_score,
                        n_jobs=-1,
                        spiketrain_mode="union",
                        verbose=False,
                        do_matching=True,
                    )
                    print(f"Saving agreement object to {agreement_save_folder}")
                    matching_obj.save_to_folder(agreement_save_folder)
                agreement_dict = self._create_agreement_dict(matching_obj)
                # add agreement columns to main dataframe
                incl_dict = self._get_subset_sorter_agreement(
                    sorting_names, agreement_dict
                )
                mask = self.metrics_df["recording"] == rec
                if col_name not in self.metrics_df.keys():
                    self.metrics_df[col_name] = 0
                self.metrics_df.loc[mask] = add_col_for_agr_subset(
                    self.metrics_df.loc[mask], incl_dict, col_name
                )

    def add_ground_truth(
        self,
        SX_gt,
        recording_name,
        well_detected_score=0.5,
        overwrite_comparison=False,
    ) -> None:
        """
        Add a ground truth sorting to the study. The sorting is provided as a SpikeInterface sorting extractor.
        The ground truth comparison is computed using the well_detected_score or read from disk if a chached comparison exists.

        Parameters
        ----------
        SX_gt: SortingExtractor
            ground truth sorting extractor
        recording_name: str
            name of recording to which ground truth belongs to (must match recording name of other sortings)
        well_detected_score: float
            score to use for ground truth comparison
        overwrite_comparison: bool
            if True, force recompute ground truth comparison when cached comparison exists
        """
        # assert (
        #     self.metrics_df is not None
        # ), "compute metrics before adding ground truth information"
        self.SX_gt = SX_gt
        self.sorter_good_ids = self._get_ground_truth_comparison(
            SX_gt, recording_name, well_detected_score, overwrite_comparison
        )
        gt_label = "gt_comp_" + str(well_detected_score).replace(".", "_")
        if gt_label not in self.metrics_df.keys():
            self.metrics_df[gt_label] = 0
        for sorter, good_unit_list in self.sorter_good_ids.items():
            self.metrics_df.loc[
                (self.metrics_df["sorter"] == sorter)
                & (self.metrics_df["recording"] == recording_name),
                gt_label,
            ] = 0
            # self.metrics_df[gt_label].mask(
            #     (self.metrics_df["sorter"] == sorter)
            #     & (self.metrics_df["recording"] == recording_name),
            #     0,
            #     inplace=True,
            # )
            self.metrics_df.loc[
                (self.metrics_df["sorter"] == sorter)
                & (self.metrics_df["sorter_unit_id"].isin(good_unit_list))
                & (self.metrics_df["recording"] == recording_name),
                gt_label,
            ] = 1
            # self.metrics_df[gt_label].mask(
            #     (self.metrics_df["sorter"] == sorter)
            #     & (self.metrics_df["sorter_unit_id"].isin(good_unit_list))
            #     & (self.metrics_df["recording"] == recording_name),
            #     1,
            #     inplace=True,
            # )

    def get_model_metrics(
        self,
        y_preds,
        target_feature=None,
        print_stats=True,
        recording=None,
        sorter=None,
        use_train_set=False,
    ) -> tuple:
        if use_train_set:
            comp_set = self.train_df
        else:
            comp_set = self.test_df
        if target_feature is None:
            y_test = comp_set[self.target_feature]
        else:
            y_test = comp_set[target_feature]
        mask = pd.Series([True] * len(comp_set), index=comp_set.index)
        if recording is not None:
            if type(recording) is list:
                mask  = mask * comp_set["recording"].isin(recording)
            else:
                mask = mask * comp_set["recording"] == recording
        if sorter is not None:
            if type(sorter) is list:
                mask  = mask * comp_set["sorter"].isin(sorter)
            else:
                mask = mask * comp_set["sorter"] == sorter
        f1 = f1_score(y_test[mask], y_preds[mask])
        recall = recall_score(y_test[mask], y_preds[mask])
        precision = precision_score(y_test[mask], y_preds[mask])
        balanced_accuracy = balanced_accuracy_score(y_test[mask], y_preds[mask], adjusted=True)
        true_positives = np.sum(y_test[mask] * y_preds[mask])
        false_positives = np.sum((1 - y_test[mask]) * y_preds[mask])
        false_negatives = np.sum(y_test[mask] * (1 - y_preds[mask]))
        true_negatives = np.sum((1 - y_test[mask]) * (1 - y_preds[mask]))
        if print_stats:
            print("f1 score:", f1)
            print("recall:   ", recall)
            print("precision:", precision)
            print("balanced_accuracy:", balanced_accuracy_score)
        results_dict = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "balanced_accuracy": balanced_accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
        }
        return results_dict
    
    def _log_transform_train_test_metrics(self) -> None:
        """
        Log transform skewed metrics
        """
        assert self.train_df is not None, "no training set available, create it first"
        assert self.test_df is not None, "no test set available, create it first"
        pd.options.mode.chained_assignment = None
        metrics, _, _, _ = self._separate_cols_to_categories()
        ordered_unique_train = self.train_df[metrics].apply(
            lambda x: np.max((x.sort_values().unique()[0], 1e-5)), axis=0
        )
        for met in list(set(self.skew_to_log) & set(metrics)):
            second_smallest = ordered_unique_train[met]
            half_second_smallest = second_smallest / 2
            mask = self.train_df[met] == 0
            self.train_df.loc[mask, met] = half_second_smallest
            mask = self.test_df[met] == 0
            self.test_df.loc[mask, met] = half_second_smallest
            self.train_df.loc[:, met] = np.log(
                self.train_df.loc[:, met].values.astype(float)
            )
            self.test_df.loc[:, met] = np.log(
                self.test_df.loc[:, met].values.astype(float)
            )

    def _fix_train_test_metrics(self, verbose=True) -> None:
        """
        Fix NaNs in metrics. This is done separately in traininbg and test data.
        """
        assert self.train_df is not None, "no training set available, create it first"
        assert self.test_df is not None, "no test set available, create it first"
        # from sklearn.impute import SimpleImputer
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        from sklearn.impute import KNNImputer

        imp = KNNImputer(n_neighbors=5, weights="distance")
        metrics, _, _, _ = self._separate_cols_to_categories()
        metrics_array = self.train_df[metrics].values
        metrics_array = imp.fit_transform(metrics_array)
        self.train_df[metrics] = metrics_array
        metrics_array = self.test_df[metrics].values.astype(float)
        metrics_array = imp.fit_transform(metrics_array)
        self.test_df[metrics] = metrics_array

    def _standard_scale_train_test_metrics(self) -> None:
        """
        Standard scale metrics
        """
        assert self.train_df is not None, "no training set available, create it first"
        assert self.test_df is not None, "no test set available, create it first"
        metrics, _, _, _ = self._separate_cols_to_categories()
        scaler = StandardScaler().set_output(transform="pandas")
        features_to_scale = metrics
        scaler.fit(self.train_df.loc[:, features_to_scale])
        self.train_df.loc[:, features_to_scale] = scaler.transform(  # type: ignore
            self.train_df[features_to_scale]
        )
        self.test_df.loc[:, features_to_scale] = scaler.transform(  # type: ignore
            self.test_df[features_to_scale]
        )

    def _clear_train_test(self, verbose=True):
        assert self.train_df is not None, "no training set available, create it first"
        assert self.test_df is not None, "no test set available, create it first"
        metrics, _, _, _ = self.get_study_info()
        all_nan = self.test_df[
            self.test_df.isna()[metrics].sum(axis=1) == len(metrics)
        ].index
        if len(all_nan) > 0:
            if verbose:
                print(
                    f"removing row(s) {all_nan} from test set because one or more metrics still are NaN"
                )
            self.test_df = self.test_df.drop(all_nan)
        all_nan = self.train_df[
            self.train_df.isna()[metrics].sum(axis=1) == len(metrics)
        ].index
        if len(all_nan) > 0:
            if verbose:
                print(
                    f"removing row(s) {all_nan} from train set because one or more metrics still are NaN"
                )
            self.train_df = self.train_df.drop(all_nan)

    def create_train_test_split(
        self,
        sorter_names=None,
        recording_names=None,
        test_size=0.5,
        random_state=42,
        standard_scale=True,
        log_transform=True,
        verbose=True,
    ):
        assert self.metrics_df is not None, "compute metrics first"
        mask = pd.Series(np.ones(len(self.metrics_df),dtype=bool))
        if sorter_names is not None:
            if type(sorter_names) is not list:
                sorter_names = [sorter_names]
            mask = mask * self.metrics_df["sorter"].isin(sorter_names)
        if recording_names is not None:
            if type(recording_names) is not list:
                recording_names = [recording_names]
            mask = mask * self.metrics_df["recording"].isin(recording_names)
        self.train_df, self.test_df = train_test_split(
            self.metrics_df[mask],
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        if verbose:
            print(
                f"training set size: {len(self.train_df)}, test size: {len(self.test_df)}"
            )
        if log_transform:
            self._log_transform_train_test_metrics()
        self._fix_train_test_metrics(verbose=verbose)
        if standard_scale:
            self._standard_scale_train_test_metrics()
        self._clear_train_test(verbose=verbose)

    def create_train_set(
        self,
        recording_names=None,
        sorter_names=None,
        train_size=1.0,
        random_state=42,
        verbose=True,
    ):
        """
        Select a subset of the training set for cross-validation
        """
        assert self.metrics_df is not None, "compute metrics first"
        if recording_names is not None:
            self.train_df = self.metrics_df[
                self.metrics_df["recording"].isin(recording_names)
            ]
        else:
            self.train_df = self.metrics_df.copy()
        if sorter_names is not None:
            self.train_df = self.train_df[self.train_df["sorter"].isin(sorter_names)]
        if train_size < 1.0:
            self.train_df, _ = train_test_split(
                self.train_df,
                train_size=train_size,
                random_state=random_state,
                shuffle=True,
                stratify=None,
            )
        if verbose:
            print(f"training set size: {len(self.train_df)}")
        self.test_df = None

    def create_test_set(
        self,
        recording_names=None,
        sorter_names=None,
        standard_scale=True,
        verbose=True,
        log_transform=True,
    ):
        """
        Select a subset of the training set for cross-validation
        """
        assert self.metrics_df is not None, "compute metrics first"
        assert (
            self.train_df is not None
        ), "cannot create test set before creating train set"
        if recording_names is not None:
            self.test_df = self.metrics_df[
                self.metrics_df["recording"].isin(recording_names)
            ]
        else:
            self.test_df = self.metrics_df.copy()
        if sorter_names is not None:
            self.test_df = self.test_df[self.test_df["sorter"].isin(sorter_names)]
        if verbose:
            print(f"test set size: {len(self.test_df)}")
        if log_transform:
            self._log_transform_train_test_metrics()
        self._fix_train_test_metrics(verbose=verbose)
        if standard_scale:
            self._standard_scale_train_test_metrics()
        # self._clear_train_test(verbose=verbose)

    def predict(self):
        assert self.model is not None, "model must be fit before predicting"
        assert (
            self.test_df is not None
        ), "no test set available, create test set before predicting"
        X_test = self.test_df[self.metrics]
        y_pred = self.model.predict(X_test)
        return y_pred

    def fit(
        self,
        metrics,
        target_feature,
        random_state=42,
        solver="saga",
        penalty="l2",
        C=1,
        predict=False,
        oversample=False,
    ):
        assert (
            self.train_df is not None
        ), "no train set available, create train set before fitting"
        assert penalty in [
            "l1",
            "l2",
            "elasticnet",
            None,
        ], "penalty must be 'l1', 'l2', 'elasticnet', or None"
        self.metrics = metrics
        self.target_feature = target_feature
        X_trn = self.train_df[metrics]
        y_trn = self.train_df[target_feature]
        # MP.test_df[metrics].isna().sum()
        if oversample:
            X_trn_resampled, y_trn_resampled = SMOTE().fit_resample(X_trn, y_trn)
            self.model = LogisticRegression(
                solver=solver, random_state=random_state, penalty=penalty, C=C
            ).fit(X_trn_resampled, y_trn_resampled)
        else:
            self.model = LogisticRegression(
                solver=solver, random_state=random_state, penalty=penalty, C=C
            ).fit(X_trn, y_trn)
            # self.model = RandomForestClassifier(
            #     random_state=random_state,
            # ).fit(X_trn, y_trn)
        if predict:
            return self.model.predict(X_trn)

    def _create_sorting_analyzer(
        self,
        sorting,
        overwrite=False,
        n_jobs=1,
        chunk_duration="1.0s", #"0.1s",
        ms_before=0.5,
        ms_after=3,
        max_spikes_per_unit=200,
        cache_folder=".",
        n_pca_components=2,
        verbose=False,
    ):
        recording_name = sorting["recording_name"]
        sorting_name = sorting["sorting_name"]
        folder = str(cache_folder) + "/waveforms_" + recording_name + "_" + sorting_name
        if os.path.exists(folder) and not overwrite:
            if verbose:
                print("loading cached waveforms from " + folder)
            sa_disk = si.load_sorting_analyzer(folder=folder)
        else:
            if verbose:
                print("caching waveforms to " + folder)
            job_kwargs = dict(n_jobs=1, chunk_duration=chunk_duration)
            sa_disk = si.create_sorting_analyzer(
                recording=sorting["recording"],
                sorting=sorting["sorting"],
                format="binary_folder",
                folder=folder,
                overwrite=True,
                # allow_unfiltered=False,
                ms_before=ms_before,
                ms_after=ms_after,
                sparse=True,
                method="radius",
                # num_spikes_for_sparsity=100,
                # radius_um=60,
                **job_kwargs,
            )
            if verbose:
                print("computing analyzer extensions")
            sa_disk.compute({
                "noise_levels":{},
                "random_spikes":{}, 
                "waveforms":{"ms_before":ms_before, "ms_after": ms_after},
                "templates":{},
                "template_similarity":{},
                "unit_locations":{},
                "principal_components":{},
                "spike_locations":{},
                "spike_amplitudes":{},
                "correlograms":{},
            }, verbose=verbose)
        return sa_disk

    def _get_ground_truth_comparison(
        self, SX_gt, recording_name, well_detected_score=0.8, overwrite_comparison=False
    ):
        # compare sortings to ground truth
        str_match_score = str(well_detected_score).replace(".", "_")
        agr_file = (
            self.study_folder
            / f"ground_truth_comparison_{recording_name}_{str_match_score}_.pkl"
        )
        if not os.path.exists(agr_file) or overwrite_comparison:
            print("computing ground truth comparison")
            sorter_good_ids = {}
            # check all sortings with correct recording
            for s in self.sortings:
                if s["recording_name"] == recording_name:
                    sorter = s["sorting_name"]
                    sr_sorting = s["sorting"]
                    # print(sr_sorting, SX_gt)
                    comp = si.compare_sorter_to_ground_truth(
                        SX_gt, sr_sorting, exhaustive_gt=True
                    )
                    well_detected = comp.get_well_detected_units(
                        well_detected_score=well_detected_score
                    )
                    sorter_good_ids[sorter] = well_detected
                    print(
                        f"{sorter}: {len(well_detected)}/{len(sr_sorting.get_unit_ids())} well-detected units, {SX_gt.get_num_units()} ground truth units"
                    )
            if len(sorter_good_ids) > 0:
                pickle.dump(sorter_good_ids, open(agr_file, "wb"))
            # with open(agr_file, 'w') as outfile:
            # json.dump(sorter_good_ids, outfile)
        else:
            print(f"loading cached ground truth comparison from {agr_file}")
            sorter_good_ids = pickle.load(open(agr_file, "rb"))
        return sorter_good_ids

    def _create_agreement_dict(self, mcmp, match_score=0.5) -> dict:
        """
        create a dictionary of agreements
        {unit_id: {sorter_name: sorter_unit_id}}
        for each unit in the agreement sorting, get the sorter unit ids for each sorter that found it
        """
        agr_1 = mcmp.get_agreement_sorting(minimum_agreement_count=1)
        global_ids = agr_1.get_unit_ids()
        agreement_dict = {}
        for i, global_unit_id in enumerate(global_ids):
            sorter_unit_ids = agr_1.get_property("unit_ids")[i]
            sorter_unit_ids = {
                key: int(value) for (key, value) in sorter_unit_ids.items()
            }  # cast float to int - bug in spikeinterface
            agreement_dict[str(global_unit_id)] = sorter_unit_ids
        return agreement_dict

    def _get_subset_sorter_agreement(self, sorter_list_include, agr_dict) -> dict:
        """
        helper for add_custom_subset_to_cumulative_df
        Given a specific recording, retrieve the agreed units for a list of sorters.
        :param sorter_list_include: list of sorter names for agreement
        :param agr_dict: dictionary created by create_agreement_dict
        :return: incl_dict: dictionary of agreed units for each sorter in sorter_list_include
        """
        incl_dict = {}
        for global_id, sorter_ids in agr_dict.items():
            sorters_for_unit = list(sorter_ids.keys())
            cond = any(item in sorters_for_unit for item in sorter_list_include)

            if cond:
                sorter_dict_include = {
                    sorter: unit_id
                    for (sorter, unit_id) in sorter_ids.items()
                    if sorter in sorter_list_include
                }
                if len(sorter_dict_include) > 1:
                    incl_dict[global_id] = sorter_dict_include
        return incl_dict

    def _separate_cols_to_categories(self, verbose=False):
        """
        Helper to extract study information from the main dataframe.

        Parameters
        ----------
        verbose: bool
            if True, print summary of study information

        Returns
        -------
        metrics: list
            list of metrics
        unit_info: list
            list of unit info columns
        agreement_targets: dict
            dictionary of agreement targets
        gt_targets: list
            list of ground truth targets
        """
        # separate columns into categories: metric column (from global list), unit info (sorter, sorter_unit_id, recording), agreement targets (dictionary of {match_score: [[sorter_set]]}), ground truth targets (list of match_score)
        # makes tight assumptions on naming conventions defined in build_dataframe.py
        assert self.metrics_df is not None, "compute metrics first"
        cols_in_df = list(self.metrics_df.columns)

        metrics = []
        unit_info = []
        agreement_targets = {}
        gt_targets = []

        for col in cols_in_df:
            if col in METRIC_LIST:
                metrics.append(col)
            elif col in UNIT_INFO:
                unit_info.append(col)
            elif col.startswith(AGREEMENT_PREFIX):
                # agreement_targets[col] = col
                components = col.split("_")
                # rec = components[1]
                match_score = float(components[1] + "." + components[2])
                sorters = []
                for i in range(3, len(components)):
                    sorters.append(components[i])
                sorters.sort()

                if match_score in agreement_targets:
                    tmp = agreement_targets[match_score]
                    tmp.append(sorters)
                    agreement_targets[match_score] = tmp
                else:
                    agreement_targets[match_score] = [sorters]
            elif col.startswith(GROUND_TRUTH_PREFIX):
                # col = 'gt_comp_0_75'
                try:
                    components = col.split("_")
                    match_score = float(components[2] + "." + components[3])
                except:
                    match_score = 1
                gt_targets.append(match_score)

        if verbose:
            print("Study summary:")
            print("--------------")
            print("metrics: " + ", ".join(map(str, metrics)))
            print("unit info: " + ", ".join(map(str, unit_info)))
            print("agreement targets:")
            for m_s, sorter_sets in agreement_targets.items():
                print(f"\tmatch score: {m_s}")
                for sorter_set in sorter_sets:
                    print("\t\t" + ", ".join(map(str, sorter_set)))
            print("ground truth targets: " + ", ".join(map(str, gt_targets)))
        return metrics, unit_info, agreement_targets, gt_targets
