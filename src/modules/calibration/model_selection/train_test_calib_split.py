from __future__ import annotations

from random import shuffle
from typing import Any, Tuple

from pandas import DataFrame, Series


def split_data_by_member_id(
    member_ids: list[str],
    train_perc: float = 0.6,
    calib_perc: float = 0.1,
    random_state: int = 42,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Splits the list of member IDs into train, calibration, and test sets based on provided percentages.

    Args:
        member_ids (List[str]): List of unique member IDs.
        train_perc (float, optional): The percentage of member IDs to allocate for training. Defaults to 0.6.
        calib_perc (float, optional): The percentage of member IDs to allocate for calibration. Defaults to 0.1.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Tuple[List[str], List[str], List[str]]: Tuple containing lists of member IDs for train, calibration, and test sets.
    """
    # Shuffle the member IDs to randomize the order
    shuffle(x=member_ids)

    total_ids: int = len(member_ids)
    train_size = int(total_ids * train_perc)
    calib_size = int(total_ids * calib_perc)

    # Assign member IDs to train, calibration, and test sets
    train_ids: list[str] = member_ids[:train_size]
    calib_ids: list[str] = member_ids[train_size : train_size + calib_size]
    test_ids: list[str] = member_ids[train_size + calib_size :]
    return train_ids, calib_ids, test_ids


def split_train_calib_test(
    samples: DataFrame,
    targets: Series,
) -> Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """
    Filters the dataset based on the member IDs allocated for train, calibration, and test sets.

    Args:
        samples (DataFrame): Input features DataFrame.
        targets (Series): Target variable Series.
        train_ids (List[str]): List of member IDs for the train set.
        calib_ids (List[str]): List of member IDs for the calibration set.
        test_ids (List[str]): List of member IDs for the test set.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]: Tuple containing filtered samples and targets for train, calibration, and test sets.
    """
    train_ids: list[str]
    calib_ids: list[str]
    test_ids: list[str]

    train_ids, calib_ids, test_ids = split_data_by_member_id(
        member_ids=samples["MemberID"].unique().tolist()
    )

    # Filter samples and targets based on member IDs
    train_samples: DataFrame = samples[samples["MemberID"].isin(values=train_ids)].drop(
        labels=["MemberID"], axis=1
    )
    calib_samples: DataFrame = samples[samples["MemberID"].isin(values=calib_ids)].drop(
        labels=["MemberID"], axis=1
    )
    test_samples: DataFrame = samples[samples["MemberID"].isin(values=test_ids)].drop(
        labels=["MemberID"], axis=1
    )

    train_targets: Series[Any] = targets.loc[train_samples.index]
    calib_targets: Series[Any] = targets.loc[calib_samples.index]
    test_targets: Series[Any] = targets.loc[test_samples.index]

    return (
        train_samples,
        calib_samples,
        test_samples,
        train_targets,
        calib_targets,
        test_targets,
    )
