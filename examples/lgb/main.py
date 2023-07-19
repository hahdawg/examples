from dataclasses import dataclass
from typing import Any, Dict, Tuple

import lightgbm as lgb
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TARGET_COL = "target"


@dataclass
class DMatrix:
    X: pd.DataFrame
    y: pd.Series


@dataclass
class TTS:
    tr: DMatrix
    val: DMatrix
    te: DMatrix


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True).frame  # pylint: disable=no-member
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]
    return X, y


def compute_tts(X: pd.DataFrame, y: pd.Series) -> TTS:
    X_tr, X_other, y_tr, y_other = train_test_split(X, y, train_size=0.7, random_state=0, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_other, y_other, train_size=0.5, random_state=0, stratify=y_other
    )
    res = TTS(
        tr=DMatrix(X_tr, y_tr),
        val=DMatrix(X_val, y_val),
        te=DMatrix(X_te, y_te)
    )
    return res


def fit_gbm(tts: TTS) -> lgb.Booster:
    dtrain = lgb.Dataset(data=tts.tr.X, label=tts.tr.y)
    dvalid = lgb.Dataset(data=tts.val.X, label=tts.val.y)
    early_stopping = lgb.early_stopping(stopping_rounds=100, first_metric_only=True)
    log = lgb.log_evaluation(period=25)
    record = lgb.record_evaluation(eval_result={})
    callbacks = [early_stopping, log, record]

    params = {
        "objective": "binary",
        "learning_rate": 2e-2,
        "max_leaves": 2,
        "min_child_weight": 5,
        "bagging_fraction": 0.9,
        "colsample_by_tree": 0.9,
        "seed": 0,
        "metric": "auc",
        "bagging_freq": 1
    }
    bst = lgb.train(
        params=params,
        num_boost_round=10000,
        train_set=dtrain,
        valid_sets=(dtrain, dvalid),
        valid_names=("tr", "val"),
        callbacks=callbacks
    )
    return bst


def main() -> None:
    X, y = load_data()
    tts = compute_tts(X=X, y=y)
    bst = fit_gbm(tts=tts)
    y_hat_te = bst.predict(tts.te.X)
    auc_te = roc_auc_score(y_true=tts.te.y, y_score=y_hat_te)
    print(f"Test AUC: {auc_te:0.4f}")


if __name__ == "__main__":
    main()
