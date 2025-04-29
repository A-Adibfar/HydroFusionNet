import torch
from torch.utils.data import Dataset


class PrecipitationDataset(Dataset):
    def __init__(self, dataframe, feature_scaler=None, diff_scaler=None, is_train=True):
        """
        dataframe: Pandas DataFrame containing all features + 'bin' target
        feature_scaler: sklearn StandardScaler or similar (for main features)
        diff_scaler: sklearn StandardScaler for differential features
        is_train: whether to load the 'bin' target or not (inference mode)
        """
        self.df = dataframe
        self.is_train = is_train
        self.feature_scaler = feature_scaler
        self.diff_scaler = diff_scaler

        # Define your feature groups (based on your list)
        self.features1 = [
            "clt1",
            "hfls1",
            "hfss1",
            "huss1",
            "pr1",
            "prc1",
            "ps1",
            "rlds1",
            "rlus1",
            "rsds1",
            "rsdsdiff1",
            "rsus1",
            "tas1",
            "uas1",
            "vas1",
        ]
        self.features2 = [
            "clt2",
            "hfls2",
            "hfss2",
            "huss2",
            "pr2",
            "prc2",
            "ps2",
            "rlds2",
            "rlus2",
            "rsds2",
            "rsdsdiff2",
            "rsus2",
            "tas2",
            "uas2",
            "vas2",
        ]
        self.features3 = [
            "clt3",
            "hfls3",
            "hfss3",
            "huss3",
            "pr3",
            "prc3",
            "ps3",
            "rlds3",
            "rlus3",
            "rsds3",
            "rsdsdiff3",
            "rsus3",
            "tas3",
            "uas3",
            "vas3",
        ]
        self.features4 = [
            "clt4",
            "hfls4",
            "hfss4",
            "huss4",
            "pr4",
            "prc4",
            "ps4",
            "rlds4",
            "rlus4",
            "rsds4",
            "rsdsdiff4",
            "rsus4",
            "tas4",
            "uas4",
            "vas4",
        ]
        self.diff_features = [
            "ps_diff_12",
            "ps_diff_13",
            "net_rad1",
            "net_rad2",
            "clt_diff_12",
            "clt_diff_13",
        ]
        self.month = ["month"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Extract feature groups
        feature1 = torch.tensor(row[self.features1].values, dtype=torch.float32)
        feature2 = torch.tensor(row[self.features2].values, dtype=torch.float32)
        feature3 = torch.tensor(row[self.features3].values, dtype=torch.float32)
        feature4 = torch.tensor(row[self.features4].values, dtype=torch.float32)

        # Differential features
        diff_features = torch.tensor(
            row[self.diff_features].values, dtype=torch.float32
        )

        # Month feature
        # month = torch.tensor(row[self.month].values, dtype=torch.float32)
        month = torch.tensor(row[self.month[0]], dtype=torch.float32)

        # Apply scaling if scalers are provided
        if self.feature_scaler:
            feature1 = torch.tensor(
                self.feature_scaler.transform(feature1.unsqueeze(0)).squeeze(0),
                dtype=torch.float32,
            )
            feature2 = torch.tensor(
                self.feature_scaler.transform(feature2.unsqueeze(0)).squeeze(0),
                dtype=torch.float32,
            )
            feature3 = torch.tensor(
                self.feature_scaler.transform(feature3.unsqueeze(0)).squeeze(0),
                dtype=torch.float32,
            )
            feature4 = torch.tensor(
                self.feature_scaler.transform(feature4.unsqueeze(0)).squeeze(0),
                dtype=torch.float32,
            )

        if self.diff_scaler:
            diff_features = torch.tensor(
                self.diff_scaler.transform(diff_features.unsqueeze(0)).squeeze(0),
                dtype=torch.float32,
            )

        if self.is_train:
            label = torch.tensor(row["bin"], dtype=torch.float32)
            return feature1, feature2, feature3, feature4, diff_features, month, label
        else:
            return feature1, feature2, feature3, feature4, diff_features, month
