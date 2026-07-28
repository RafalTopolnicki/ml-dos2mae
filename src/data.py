from random import gauss

import pandas as pd
import numpy as np
from scipy import interpolate
from tqdm import tqdm
from copy import deepcopy

import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from src.consts import STANDARD_CONST_DOS
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

from src.consts import DOS_INTERPOLATION_GRID_PTS, PT_COLUMN_NAME


def read_occupancy(path, return_sum=False, keep_d_only=False, use_grad=False):
    df = pd.DataFrame(np.loadtxt(path))
    if keep_d_only:
        # df = df.iloc[:, [0, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18]]
        df = df.iloc[:, [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18]]
    df.columns = ["E"] + [f"dos_{i}" for i in range(df.shape[1] - 1)]
    df = df[df["E"] < 7]
    ## apply gradient
    if use_grad:
        for col in df.columns:
            if col != "E":
                df[col] = np.gradient(df[col])
    if return_sum:
        df = pd.concat([df["E"], df.drop("E", axis=1).sum(axis=1)], axis=1)
        df.columns = ["E", "dos"]
    return df


def read_occupancy_nitrogen(path, return_sum=False, use_grad=False):
    df = pd.DataFrame(np.loadtxt(path))
    # df = df.iloc[:, [0, 1, 2, 3, 4, 10, 11, 12, 13]]
    # keep only p-orbitals
    df = df.iloc[:, [0, 2, 3, 4, 11, 12, 13]]
    df.columns = ["E"] + [f"dosN_{i}" for i in range(df.shape[1] - 1)]
    df = df[df["E"] < 7]
    ## apply gradient
    if use_grad:
        for col in df.columns:
            if col != "E":
                df[col] = np.gradient(df[col])
    if return_sum:
        df = pd.concat([df["E"], df.drop("E", axis=1).sum(axis=1)], axis=1)
        df.columns = ["E", "dos"]
    return df


# def read_doscar(path, return_sum=False, keep_d_only=False):
#     df = pd.read_csv(path, sep="  ", skiprows=1, engine="python", header=None)
#     # https://www.vasp.at/wiki/index.php/DOSCAR
#     #      1  2    3    4    5     6     7        8     9
#     # UP   s, p_y, p_z, p_x, d_xy, d_yz, d_z2-r2, d_xz, d_x2-y2
#     #      10 11   12   13   14    15    16       17    18
#     # DOWN s, p_y, p_z, p_x, d_xy, d_yz, d_z2-r2, d_xz, d_x2-y2
#     if keep_d_only:
#         # df = df.iloc[:, [0, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18]]
#         df = df.iloc[:, [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18]]
#     df.columns = ["E"] + [f"dos_{i}" for i in range(df.shape[1] - 1)]
#     df = df[df["E"] < 7]
#     if return_sum:
#         df = pd.concat([df["E"], df.drop("E", axis=1).sum(axis=1)], axis=1)
#         df.columns = ["E", "dos"]
#     return df

def read_doscar(path, return_sum=False, keep_d_only=False):
    df = pd.read_csv(path, sep="  ", skiprows=1, engine="python", header=None)
    # https://www.vasp.at/wiki/index.php/DOSCAR
    #      1    2      3      4     5     6     7     8
    #     su,  sd,  p_yd,  p_yu, p_zu, p_zd, p_xu, p_xd
    #           9      10      11      12      13      14      15      16         17         18
    #     d_{xy}u d_{xy}d d_{yz}u d_{yz}d d_{z2}u d_{z2}d d_{xz}u d_{xz}d d_{x2-y2}u d_{x2-y2}u
    if keep_d_only:
        df = df.iloc[:, [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    df.columns = ["E"] + [f"dos_{i}" for i in range(df.shape[1] - 1)]
    df = df[df["E"] < 7]
    if return_sum:
        df = pd.concat([df["E"], df.drop("E", axis=1).sum(axis=1)], axis=1)
        df.columns = ["E", "dos"]
    return df


def interpolate_spectrum(
    df, energy_window_low, energy_window_high, energy_window_interpolation_pts=DOS_INTERPOLATION_GRID_PTS, keepE=False
):
    energy_points = np.linspace(energy_window_low, energy_window_high, energy_window_interpolation_pts)
    df_out = {"E": energy_points}
    for col in df.columns:
        interpolator = interpolate.interp1d(df["E"], df[col], fill_value="extrapolate")
        y = interpolator(energy_points)
        df_out[col] = y
    df = pd.DataFrame(df_out)
    if keepE == False:
        df = df.drop(["E"], axis=1)
    return df


def smooth_spectrum(df, sigma):
    df_out = df.copy()
    for col in df_out.columns:
        if col != "E":
            df_out[col] = gaussian_filter1d(df_out[col], sigma=sigma)
    return df_out

def add_random_gaussians(
    df: pd.DataFrame,
    n_gaussians_per_column: int = 1,
    width_range: tuple[float, float] = (0.01, 0.2),
    height_fraction: float = 0.5,
    excluded_energy_window: tuple[float, float] = (-2.0, 2.0),
    random_state: int = 42,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add random Gaussian peaks to all columns named dos_0, dos_1, ..., dos_N.

    Gaussians are centered only outside the excluded energy window:
        E < -2 or E > 2 by default.

    Parameters
    ----------
    df:
        DataFrame containing column 'E' and numerical DOS columns.
    n_gaussians_per_column:
        Number of random Gaussians to add to each DOS column.
    width_range:
        Random Gaussian sigma is drawn uniformly from this range.
    height_fraction:
        Maximum Gaussian height is this fraction of the column maximum.
        Actual height is drawn uniformly from 0 to height_fraction * max(column).
    excluded_energy_window:
        Energy interval where Gaussian centers are not allowed.
    random_state:
        Seed for reproducibility.
    inplace:
        If True, modify df directly. Otherwise return a modified copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with added Gaussian peaks.
    """

    rng = np.random.default_rng(random_state)

    out = df if inplace else df.copy()

    E = out["E"].to_numpy(dtype=float)

    dos_cols = [
        col for col in out.columns
        if col.startswith("dos_")
    ]

    Emin_excl, Emax_excl = excluded_energy_window

    allowed_mask = (E < Emin_excl) | (E > Emax_excl)
    allowed_E = E[allowed_mask]

    if len(allowed_E) == 0:
        print('No noise added')
        return df

    for col in dos_cols:
        y = out[col].to_numpy(dtype=float)

        col_max = np.max(y)
        if col_max == 0:
            # Avoid adding zero-height Gaussians to an all-zero column.
            continue

        for _ in range(n_gaussians_per_column):
            center = rng.choice(allowed_E)
            sigma = rng.uniform(*width_range)
            height = rng.uniform(0.0, height_fraction * col_max)

            gaussian = height * np.exp(-0.5 * ((E - center) / sigma) ** 2)

            # Add Gaussian values only where E is outside [-2, 2]
            y[allowed_mask] += gaussian[allowed_mask]

        out[col] = y

    return out

def process_database(
    path,
    dataset_path,
    energy_window_low,
    energy_window_high,
    energy_window_interpolation_pts,
    smooth_sigma=0.0,
    shuffle=True,
    keep_d_only=False,
    keepE=False,
    split_at_fermi=False,
    input_occupancy=False,
    add_nitrogen=False,
    use_grad=False,
    use_pt=False,
    n_limit=0,
    target_label="MAE",
    noise_height_fraction=0.0,
):
    database = pd.read_csv(path)
    if n_limit > 0:
        database = database.head(n_limit)
    if "AtomID_bottom" not in database.columns:
        database["AtomID_bottom"] = 50
    if "AtomID_top" not in database.columns:
        database["AtomID_top"] = 51

    columns_to_drop = []
    if target_label == "MAE":
        columns_to_drop = [c for c in database.columns if "target_" in c and "MAE" not in c]
    if target_label == "ESOC_bottom":
        columns_to_drop = [c for c in database.columns if "target_" in c and "target_ecos_bottom" not in c]
    if target_label == "ESOC_top":
        columns_to_drop = [c for c in database.columns if "target_" in c and "target_ecos_top" not in c]
    if target_label == "ESOC":
        columns_to_drop = [c for c in database.columns if "target_" in c and "target_ecos" not in c]
    if target_label == "MAE+ESOC":
        columns_to_drop = [c for c in database.columns if "target_" in c and "target_ecos" not in c and "MAE" not in c]
    if target_label == "DORBITALS":
        columns_to_drop = [
            c
            for c in database.columns
            if "target_ecos" in c or "MAE" in c or "target_top_p" in c or "target_bottom_p" in c
        ]
    if target_label == "MAE+ESOC+DORBITALS":
        columns_to_drop = [c for c in database.columns if "target_top_p" in c or "target_bottom_p" in c]
    columns_to_drop += [
        "target_top_d_xy:z2-r2",
        "target_top_d_z2-r2:x2-y2",
        "target_bottom_d_xy:z2-r2",
        "target_bottom_d_z2-r2:x2-y2",
    ]
    database = database.drop(columns_to_drop, axis=1)

    # scale all columns
    target_cols = [col for col in database.columns if "target" in col]
    target_cols = sorted(target_cols)
    print(f"III Target col order: {target_cols}")

    if shuffle:
        database = database.sample(frac=1).reset_index(drop=True)

    features_columns = [c for c in database.columns if "Feature" in c]
    assert PT_COLUMN_NAME in features_columns

    doss = []
    for _, row in tqdm(database.iterrows(), total=len(database)):
        path = row["PATH"]
        id_local = str(row["ID_local"]).zfill(3)
        target = row[target_cols]
        if input_occupancy:
            atom_id = row["Atom_bottom"]
            doscar_path = os.path.join(dataset_path, path, f"{id_local}_SR_Occupancies_{atom_id}.txt")  # FIXTHIS
            dos1 = read_occupancy(doscar_path, return_sum=False, keep_d_only=keep_d_only, use_grad=use_grad)  # BOTTOM
        else:
            atom_id = row["AtomID_bottom"]
            doscar_path = os.path.join(dataset_path, path, f"{id_local}_SR_DOSCAR_{atom_id}")  # FIXTHIS
            dos1 = read_doscar(doscar_path, return_sum=False, keep_d_only=keep_d_only)  # BOTTOM

        if input_occupancy:
            atom_id = row["Atom_top"]
            doscar_path = os.path.join(dataset_path, path, f"{id_local}_SR_Occupancies_{atom_id}.txt")  # FIXTHIS
            dos2 = read_occupancy(doscar_path, return_sum=False, keep_d_only=keep_d_only, use_grad=use_grad)  # TOP
        else:
            atom_id = row["AtomID_top"]
            doscar_path = os.path.join(dataset_path, path, f"{id_local}_SR_DOSCAR_{atom_id}")  # FIXTHIS
            dos2 = read_doscar(doscar_path, return_sum=False, keep_d_only=keep_d_only)  # TOP

        if input_occupancy and add_nitrogen:
            occupancy_path = os.path.join(dataset_path, path, f"{id_local}_SR_Occupancies_N3.txt")  # FIXTHIS
            dos_nitrogen = read_occupancy_nitrogen(occupancy_path, return_sum=False, use_grad=use_grad)
            # append the nitogen data to first atom
            dos_nitrogen.drop(["E"], axis=1, inplace=True)
            dos1 = pd.concat([dos1, dos_nitrogen], axis=1)

        if smooth_sigma > 0:
            dos1 = smooth_spectrum(dos1, sigma=smooth_sigma)
            dos2 = smooth_spectrum(dos2, sigma=smooth_sigma)


        dos1 = interpolate_spectrum(
            dos1,
            energy_window_low=energy_window_low,
            energy_window_high=energy_window_high,
            energy_window_interpolation_pts=energy_window_interpolation_pts,
            keepE=True,
        )
        dos2 = interpolate_spectrum(
            dos2,
            energy_window_low=energy_window_low,
            energy_window_high=energy_window_high,
            energy_window_interpolation_pts=energy_window_interpolation_pts,
            keepE=True,
        )
        # adding noise
        if noise_height_fraction > 1e-5:
            #print(f'&&&&&&& ADD NOISE {noise_height_fraction}')
            dos1 = add_random_gaussians(dos1, n_gaussians_per_column=10, width_range=(0.01, 0.2), height_fraction=noise_height_fraction, excluded_energy_window=(-0.01, 0.01))
            dos2 = add_random_gaussians(dos2, n_gaussians_per_column=10, width_range=(0.01, 0.2), height_fraction=noise_height_fraction, excluded_energy_window=(-0.01, 0.01))

        if split_at_fermi:
            dos1_below = dos1[dos1["E"] <= 0].reset_index(drop=True)
            dos1_above = dos1[dos1["E"] >= 0].reset_index(drop=True)
            dos2_below = dos2[dos2["E"] <= 0].reset_index(drop=True)
            dos2_above = dos2[dos2["E"] >= 0].reset_index(drop=True)

            dos2_above.drop(["E"], axis=1, inplace=True)
            dos2_below.drop(["E"], axis=1, inplace=True)
            dos1_above.drop(["E"], axis=1, inplace=True)
            if keepE == False:
                dos1_below.drop(["E"], axis=1, inplace=True)
            # reverse the above dataframes
            dos1_above = dos1_above.iloc[::-1]
            dos2_above = dos2_above.iloc[::-1]
            dos = np.hstack([dos1_below, dos1_above, dos2_below, dos2_above])
        else:
            dos2 = dos2.drop("E", axis=1)
            if keepE == False:
                dos1 = dos1.drop("E", axis=1)
            dos = np.hstack([dos1, dos2])
        identifier = f"{path}_{id_local}"

        if use_pt is False:
            row[PT_COLUMN_NAME] = 0.0
        # extract features
        features = np.array(row[features_columns].to_list()).astype(np.float16)

        assert len(target) == len(target_cols)

        # DOS
        # BOTTOM_BELOW BOTTOM_ABOVE TOP_BELOW TOP_ABOVE
        doss.append(
            {
                "dos": dos.astype(np.float32) / STANDARD_CONST_DOS,  # THIS DIVIDES THE ENERGY AS WELL
                "target": np.float32(target),
                "identifier": identifier,
                "features": features,
            }
        )
    return doss, features.shape[0], len(target_cols), target_cols


def process_database_pt(
    path,
    shuffle=True,
    n_limit=0,
):
    database = pd.read_csv(path)
    if n_limit > 0:
        database = database.head(n_limit)

    if shuffle:
        database = database.sample(frac=1).reset_index(drop=True)

    return database


def train_val_test_split(X, cv_folds: int, which_folds, train_on_whole_data: bool = False):
    # assert len(X) == len(y)
    if train_on_whole_data:
        # fold_id, X_train, X_val, X_test, y_train, y_val, y_test
        yield 0, X, [np.nan], [np.nan]
    else:
        kf = KFold(n_splits=cv_folds)
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(X)):
            if fold_id in which_folds:
                train_idx, val_idx = train_test_split(trainval_idx, test_size=1 / (cv_folds - 1))
                assert len(set(train_idx).intersection(val_idx)) == 0, "Overlap between train and val"
                assert len(set(train_idx).intersection(test_idx)) == 0, "Overlap between train and test"
                assert len(set(val_idx).intersection(test_idx)) == 0, "Overlap between val and test"
                X_train = [X[idx] for idx in train_idx]
                X_val = [X[idx] for idx in val_idx]
                X_test = [X[idx] for idx in test_idx]
                yield fold_id, X_train, X_val, X_test


def train_val_test_split_pt(X, cv_folds: int, which_folds):
    kf = KFold(n_splits=cv_folds)
    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(X)):
        if fold_id in which_folds:
            train_idx, val_idx = train_test_split(trainval_idx, test_size=1 / (cv_folds - 1))
            X_train = [X.iloc[idx] for idx in train_idx]
            X_val = [X.iloc[idx] for idx in val_idx]
            X_test = [X.iloc[idx] for idx in test_idx]
            yield fold_id, X_train, X_val, X_test


class MAEDOSDataset(Dataset):
    def __init__(self, list_data_dict, scaler=None):
        self.list_data_dict = deepcopy(list_data_dict)

        if scaler is None:
            print("Fitting scaler")
            targets = [np.asarray(x["target"]) for x in self.list_data_dict]
            targets = np.array(targets)
            self.scaler = StandardScaler()
            self.scaler.fit(targets)
        else:
            print("Using predefined scaler")
            self.scaler = deepcopy(scaler)

        for x in self.list_data_dict:
            target = np.asarray(x["target"])
            x["target"] = self.scaler.transform(target.reshape(1, -1)).reshape(-1)


    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        return (
            self.list_data_dict[idx]["dos"],
            self.list_data_dict[idx]["target"],
            self.list_data_dict[idx]["identifier"],
            self.list_data_dict[idx]["features"],
        )
