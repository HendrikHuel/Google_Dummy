#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:41:11 2022

@author: huelsbusch
"""
import sys

import dask

sys.path.insert(0, "src")

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from project.get_project_root import get_project_root

pd.set_option("mode.chained_assignment", None)

ROOT = get_project_root()


###########################
## Read and preprocess data
###########################
def preprocess_data(df: pd.DataFrame):
    """
    Some transformation and data cleaning -- not computational heavy.

    Parameters
    ----------
    df : pd.DataFrame
        The .csv data.

    Returns
    -------
    df : pd.DataFrame
        The .csv's data transformed and cleaned.

    """
    ## do something
    return df


def save_preprocessed(df: pd.DataFrame, fname: str, ek: int = 1, root: Path = ROOT):
    """
    Save preprocessed data to data/intermediate/preprocessed/ek{str(ek).zfill(3)}.
    Parameters
    ----------
    df : pd.DataFrame
        The .csv's data transformed and cleaned.
    fname : Path
        The the files name.
    ek : int, optional
        The EK group. The default is 1.
    root : Path, optional
        The projects root. The default is ROOT.

    Returns
    -------
    None.

    """
    prep_dir = root / "data" / "intermediate" / "preprocessed" / f"ek{str(ek).zfill(3)}"

    df.reset_index(drop=True, inplace=True)
    df.to_feather(prep_dir / fname)


@dask.delayed
def read_preprocess_write(f: Path, ek: int = 1):
    """
    Read and preprocess the data. Saves the preprocessed data to
    data/intermediate/preprocessed/ek{str(ek).zfill(3)}.

    Parameters
    ----------
    f : Path
        The path to the .csv file.

    Returns
    -------
    df : pd.DataFrame
        The .csv's data transformed and cleaned.

    """
    df = pd.read_csv(f, sep=";", decimal=".")
    df = preprocess_data(df)
    save_preprocessed(df, fname=f.name.replace(".csv", ".feather"), ek=ek)
    return df


def run_preprocessing(ek: int = 1, root: Path = ROOT):
    """
    Preprocess .csv data in parallel using dask and save the results as chunks
    as .feather's to data/intermediate/preprocessed/ek{str(ek).zfill(3)}.

    Parameters
    ----------
    ek : int, optional
        The EK group. The default is 1.
    root : Path, optional
        The projects root. The default is ROOT.

    Returns
    -------
    df_pre : pd.DataFrame
        A concatenated version of the preprocessed data.

    """
    input_dir = root / "data" / "input" / f"ek{str(ek).zfill(3)}"
    files = list(input_dir.glob("file*.csv"))

    df_pre = [read_preprocess_write(f) for f in files]
    df_pre = dask.compute(df_pre)[0]

    df_pre = pd.concat(df_pre)

    return df_pre


###############
## Add features
###############


def read_preprocessed(ek: int = 1, root: Path = ROOT):
    """
    Read the preprocessed data from data/intermediate/preprocessed/ek{str(ek).zfill(3)}
    in parallel.

    Parameters
    ----------
    ek : int, optional
        The EK group. The default is 1.
    root : Path, optional
        The projects root. The default is ROOT.

    Returns
    -------
    df_pre : pd.DataFrame
        A concatenated Version of the preprocessed data.

    """
    prep_dir = root / "data" / "intermediate" / "preprocessed" / f"ek{str(ek).zfill(3)}"

    @dask.delayed
    def read_feather(f):
        "A dask wraper to read .feather files in prallel."
        return pd.read_feather(f)

    files = prep_dir.glob("*.feather")

    df_pre = [read_feather(f) for f in files]
    df_pre = dask.compute(df_pre)[0]
    df_pre = pd.concat(df_pre)

    return df_pre


@dask.delayed
def add_random_feature(df: pd.DataFrame):
    """
    Add a computational heavy feature.

    Parameters
    ----------
    df : pd.DataFrame
        The data.

    Returns
    -------
    df : pd.DataFrame
        The data with the added feature.
    """
    df["RANDOM_FEATURE"] = np.random.randn(len(df))
    return df


def add_pandas_feature(df: pd.DataFrame):
    """
    Add a pandas based feature. In this case a rolling mean is added.

    Parameters
    ----------
    df : pd.DataFrame
        The data.

    Returns
    -------
    df : TYPE
        The data with the rolling mean as an added feature.

    """
    df.sort_values("MATNR", inplace=True)
    df["ABSATZ_ROLLING"] = (
        df.groupby(["MATNR"])["L_ABSATZ2"].rolling(2).mean().fillna(0)
    ).to_numpy()
    return df


def add_all_features(df: pd.DataFrame):
    """
    Add all features.

    Parameters
    ----------
    df : pd.DataFrame
        The data.

    Returns
    -------
    df : TYPE
        The data with the added features.

    """

    df = add_pandas_feature(df)

    chunks = [df.query(f"WERKS == {w}") for w in df["WERKS"].unique()]

    chunks = [add_random_feature(df_c) for df_c in chunks]

    chunks = dask.compute(chunks)[0]

    df = pd.concat(chunks)

    return df


def save_engineered(
    df: pd.DataFrame, ek: int = 1, fname: str = "engineered_data", root: Path = ROOT,
):
    """
    Save the engineered data to .feather in
    "data/intermediate/engineered/f"ek{str(ek).zfill(3)}".

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    ek : int, optional
        The EK group. The default is 1.
    fname : str, optional
        The .feather file's name. The default is "engineered_data".
    root : Path, optional
        The projects root. The default is ROOT.

    Returns
    -------
    None.

    """
    int_dir = root / "data" / "intermediate" / "engineered" / f"ek{str(ek).zfill(3)}"

    df.reset_index(inplace=True, drop=True)

    df.to_feather(int_dir / f"{fname}.feather")


def run_engineering(ek: int = 1):
    """
    Run the feature engineering pipeline.

    Parameters
    ----------
    ek : int, optional
        The EK group. The default is 1.

    Returns
    -------
    df_eng : pd.DataFrame
        The engineered data.

    """
    df_eng = read_preprocessed(ek=ek)
    df_eng = add_all_features(df_eng)
    save_engineered(df_eng)

    return df_eng


#######################
## Train and save model
#######################


def read_engineered(ek: int = 1, fname: str = "engineered_data", root: Path = ROOT):
    """
    Read the pre-engineered data from
    "data/intermediate/engineered/f"ek{str(ek).zfill(3)}".

    Parameters
    ----------
    ek : int, optional
        The EK group. The default is 1.
    fname : str, optional
        The .feather file's name. The default is "engineered_data".
    root : Path, optional
        The projects root. The default is ROOT.

    Returns
    -------
    df_eng : pd.DataFrame
        The engineered data.

    """

    int_dir = root / "data" / "intermediate" / "engineered" / f"ek{str(ek).zfill(3)}"
    df = pd.read_feather(int_dir / f"{fname}.feather")

    return df


def train_lgb(
    df_eng: pd.DataFrame,
    label: str = "L_ABSATZ3",
    feature_list: list = ["L_ABSATZ2", "ABSATZ_ROLLING", "RANDOM_FEATURE"],
    categorical_features: list = ["WERKS"],
    params: dict = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.85,
        "metric": ["mae", "rmse", "tweedie"],
        "num_iterations": 10,
    },
):
    """
    Train the lgb model.

    Parameters
    ----------
    df_eng : pd.DataFrame
        The engineered data.
    label : str, optional
        The target for the fit. The default is "L_ABSATZ3".
    feature_list : list, optional
        The non categorical-features. The default is ["L_ABSATZ2", "ABSATZ_ROLLING", "RANDOM_FEATURE"].
    categorical_features : list, optional
        The categorical-features. The default is ["WERKS"].
    params : dict, optional
        The model's parameters.

    Returns
    -------
    lgbm : lgb.booster
        A fitted lgb model.

    """

    dtrain = lgb.Dataset(
        data=df_eng[feature_list + categorical_features],
        label=df_eng[label],
        feature_name=feature_list + categorical_features,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    lgbm = lgb.train(params, dtrain,)

    return lgbm


def save_lgb(lgbm, ek: int = 1, model_name: str = "test_model", root: Path = ROOT):
    """
    Save the lgb as pickle in "data/intermediate/models/f"ek{str(ek).zfill(3)}".


    Parameters
    ----------
    lgbm : lgb.booster
        A fitted lgb model.
    ek : int, optional
        The EK group. The default is 1.
    model_name : str, optional
        The model's name. The default is "test_model".
    root : Path, optional
        The projects root. The default is ROOT.

    Returns
    -------
    None.

    """
    model_dir = root / "data" / "intermediate" / "models" / f"ek{str(ek).zfill(3)}"

    with (model_dir / f"{model_name}.pkl").open("wb") as f:
        pickle.dump(lgbm, f)


def run_training_pipe(ek: int = 1):
    """
    Run the training pipline. Reads the engineered data and estimates a lgb model.

    Parameters
    ----------
    ek : int, optional
        The EK group. The default is 1.

    Returns
    -------
    lgbm : lgb.booster
        A fitted lgb model.

    """
    df_eng = read_engineered(ek=ek)
    lgbm = train_lgb(df_eng)
    save_lgb(lgbm, ek=ek)

    return lgbm


#########################
### For the main function
#########################


def preprocess_and_train(
    ek: int = 1, preprocess: bool = True, engineer: bool = True, train: bool = True,
):
    """
    Controlls the feature engineering and model estimation pipelines for an "ek".
    In our real world application we have more then ten "ek"s.

    The pipline is constructed such that: First, all steps depend squential on
    each other in terms of data in and output and, second, all steps can be
    run seperately.

    In total the pipline consists of three steps:
        1. Preprocessing of the .csv files. In our real world application only the .csv's for the most recent date have to be processed.
        2. Pre-calculation of features.
        3. Model estimation

    For our real world application the preprocessing and pre-calculation of
    features runs Thursdays. First the preprocessing updates the newest batch
    of incoming .csv's and there will be a new batch for each "ek". Then the
    feature engineering is triggered. The model estimation is done over the
    weekend since it take up to 39 hours. At the moment all the model estimation
    is done squentialy for each "ek". An the maximal runtime here is 11 hours for a single "ek".

    Each "ek" has several hundreds of these input .csv files in total. On very infrequent
    basis the preprocessing has to be re-run for all files. For example, in te case when a bug is discovered or the preprocessing was
    enhanced. For our biggest "ek"s the processing of all .csv's can take up to four hours using
    parallelisation where 4 files are processed at once. This is why the
    preprocessing in this dummy app uses dask, because sometimes we need it.


    The flow chart of our process looks as follows:

        file[x].csv's -> <preprocessing> -> file[x].feather's -> <engineering> -> engineered_data.feather -> <model training> -> test_model.pkl

    We use the following data structure.
    The input data for the project is stored at
        0. data/input/ek{str(ek).zfill(3)}
    in .csv format.

    For each of the afromentioned steps, the input data is processed and stored
    during the run of the pipline at different locations as follows:
        1. preprocessed data: data/intermediate/preprocessed/ek{str(ek).zfill(3)}
        2. engineered data: data/intermediate/engineered/f"ek{str(ek).zfill(3)}
        3. models: "data/intermediate/models/f"ek{str(ek).zfill(3)}"

    The models are then used to make predictions at hourly frequency for all "ek"s. This step is omitted in this dummy app to keep things simple.

    Parameters
    ----------
    ek : int, optional
        The EK group. The default is 1.
    preprocess : bool, optional
        Run the preprocessing. The default is True.
    engineer : bool, optional
        Run the feature engineering. The default is True.
    train : bool, optional
        Train the model. The default is True.

    Returns
    -------
    None.

    """

    if preprocess:
        run_preprocessing(ek=ek)
    if engineer:
        run_engineering(ek=ek)
    if train:
        run_training_pipe(ek=ek)


if __name__ == "__main__":

    df_pre = run_preprocessing(ek=1)

    df_eng = run_engineering(ek=1)

    lgbm = run_training_pipe(ek=1)
