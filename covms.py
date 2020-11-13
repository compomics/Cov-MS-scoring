import numpy as np
import pandas as pd


class FeatureImport:
    def __init__(self, filename=None):
        self.filename = filename

        self.skyline_export = None
        self.sample_df = None
        self.precursor_df = None
        self.transition_df = None
        self.features = None

        self.ion_type_mapping = {"b": 0, "y": 1}
        self.swab_mapping = {"UTM": 0, "eSwab": 1}

    def _read_skyline_export_csv(self):
        self.skyline_export = pd.read_csv(self.filename, sep=";", decimal=",")

    def _preprocess_skyline_export(self):
        # Fill missing values
        self.skyline_export.dropna(subset=["File Name"], axis=0, inplace=True)
        fillna_cols = [
            "Max Fwhm",
            "Library Dot Product",
            "Best Retention Time",
            "Library Rank",
            "Library Intensity",
        ]
        for col in fillna_cols:
            self.skyline_export[col].fillna(0.0, inplace=True)

        if self.skyline_export.isna().any().any():
            raise ValueError("Unexpected missing values found in Skyline export.")

        # Map string values to integers
        self.skyline_export["Swab ID"] = self.skyline_export["Swab"].map(
            self.swab_mapping
        )

        self.skyline_export["Fragment Number"] = (
            self.skyline_export["Fragment Ion"]
            .str.replace("y", "")
            .str.replace("b", "")
            .astype(int)
        )

        self.skyline_export["Fragment Ion Type"] = self.skyline_export[
            "Fragment Ion Type"
        ].map(self.ion_type_mapping)

        self.skyline_export["Patient_Sample"] = self.skyline_export["Patient_Sample"].astype(
            int
        )

        # Parse percentage to float
        self.skyline_export["Area Normalized"] = (
            self.skyline_export["Area Normalized"]
            .str.replace("%", "")
            .str.replace(",", ".")
            .astype(float)
        )

        rt = self.skyline_export["Retention Time"]
        start = self.skyline_export["Start Time"]
        end = self.skyline_export["End Time"]
        rt_diff = (rt - (start + (end - start) / 2)).abs()
        self.skyline_export["Retention Time Deviation"] = rt_diff

    def _split_skyline_export(self):
        sample_id_cols = ["File Name"]
        sample_cols = ["Swab ID"]

        precursor_id_cols = ["Peptide Modified Sequence", "Precursor Charge"]
        precursor_cols = [
            "Best Retention Time",
            "Total Area Fragment",
            "Library Dot Product",
            "Total Background",
            "Max Fwhm",
        ]

        transition_cols = [
            "Retention Time",
            "Retention Time Deviation",
            "Fwhm",
            "Area",
            "Background",
            "Height",
            "Library Rank",
            "Library Intensity",
            "Area Normalized",
        ]

        # Set sample and precursor indexes
        self.sample_index = (
            self.skyline_export[sample_id_cols]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "Sample Index"})
        )
        self.precursor_index = (
            self.skyline_export[precursor_id_cols]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "Precursor Index"})
        )

        self.skyline_export = self.skyline_export.merge(self.sample_index)
        self.skyline_export = self.skyline_export.merge(self.precursor_index)

        # Restructure
        self.sample_df = self.skyline_export[
            ["Sample Index"] + sample_cols
        ].drop_duplicates()

        self.precursor_df = self.skyline_export[
            ["Sample Index", "Precursor Index"] + precursor_cols
        ].drop_duplicates()

        self.transition_df = self.skyline_export[
            ["Sample Index", "Precursor Index"] + transition_cols
        ].drop_duplicates()

        assert len(self.sample_df) == len(
            self.sample_index
        ), "Unexpected number of rows in sample_df."
        assert len(self.precursor_df) == len(self.precursor_index) * len(
            self.sample_index
        ), "Unexpected number of rows in precursor_df."
        assert len(self.transition_df) == len(
            self.skyline_export
        ), "Unexpected number of rows in transition_df"

    def _process_precursor_df(self):
        pdf = self.precursor_df
        log_cols = ["Total Background", "Total Area Fragment", "Max Fwhm"]

        for col in log_cols:
            pdf[col + " (log)"] = np.log10(pdf[col] + 1)
        pdf.drop(columns=log_cols, inplace=True)

        pdf["Total Area / Background (log)"] = (
            pdf["Total Area Fragment (log)"] / pdf["Total Background (log)"]
        )

    def _process_transition_df(self):
        tdf = self.transition_df
        log_cols = [
            "Background",
            "Area",
            "Area Normalized",
            "Height",
            "Library Intensity",
            "Fwhm",
        ]

        for col in log_cols:
            tdf[col + " (log)"] = np.log10(tdf[col] + 1)
        tdf.drop(columns=log_cols, inplace=True)

        tdf["Area (log) / Background (log)"] = (
            tdf["Area (log)"] / tdf["Background (log)"]
        )
        tdf["Area Normalized (log) / Background (log)"] = (
            tdf["Area Normalized (log)"] / tdf["Background (log)"]
        )

    def _reshape_features(self, exclusion_list=None):
        if not exclusion_list:
            exclusion_list = []

        def reshape_df_generator(df, id_col, new_id_col, id_exclusion_list):
            for t_id in df[id_col].unique():
                if t_id in id_exclusion_list:
                    continue
                tmp_df = (
                    df[df[id_col] == t_id]
                    .copy()
                    .set_index(new_id_col)
                    .drop(id_col, axis=1)
                )
                tmp_df.columns = [col + " " + str(t_id) for col in tmp_df.columns]
                yield tmp_df

        # Reshape precursors
        precursor_df_reshaped = pd.concat(
            reshape_df_generator(
                self.precursor_df, "Precursor Index", "Sample Index", []
            ),
            axis=1,
        )
        assert (
            not precursor_df_reshaped.isna().any().any()
        ), "Missing values found in reshaped transition DataFrame."

        # Reshape transitions
        self.transition_df["id"] = (
            self.skyline_export["Precursor Index"].astype(str)
            + "_"
            + self.skyline_export["Fragment Ion"].astype(str)
        )
        id_exclusion_list = ["7_y10", "11_y3"]
        transition_df_reshaped = pd.concat(
            reshape_df_generator(
                self.transition_df, "id", "Sample Index", id_exclusion_list
            ),
            axis=1,
        )
        assert (
            not transition_df_reshaped.isna().any().any()
        ), "Missing values found in reshaped transition DataFrame."

        self.features = pd.concat(
            [
                self.sample_df.set_index("Sample Index"),
                precursor_df_reshaped,
                transition_df_reshaped,
            ],
            axis=1,
        )
        self.features.replace(np.inf, 0, inplace=True)

    def from_skyline_export(self, filename=None):
        if filename:
            self.filename = filename

        self._read_skyline_export_csv()
        self._preprocess_skyline_export()
        self._split_skyline_export()
        self._process_precursor_df()
        self._process_transition_df()
        self._reshape_features()

    @property
    def feature_names(self):
        return self.features.columns

    @property
    def patient_sample_mapping(self):
        return (
            self.skyline_export[["Patient_Sample", "Sample Index"]]
            .drop_duplicates()
            .set_index("Patient_Sample")["Sample Index"]
            .to_dict()
        )


class TargetImport:
    def __init__(self, patient_sample_mapping, filename=None):
        self.patient_sample_mapping = patient_sample_mapping
        self.filename = filename

        self.pcr_result = None
        self.mean_ct = None
        self.outcome = None

    def read_csv(self, filename=None):
        if filename:
            self.filename = filename

        self.pcr_result = pd.read_csv(self.filename, sep=";")
        self.pcr_result["ct_mean"] = self.pcr_result[["ct1", "ct2", "ct3"]].mean(axis=1)
        self.pcr_result["Sample Index"] = self.pcr_result["patient"].map(
            self.patient_sample_mapping
        )
        self.pcr_result.set_index("Sample Index", inplace=True)
        self.mean_ct = self.pcr_result["ct_mean"]
        self.outcome = self.pcr_result["outcome"]
