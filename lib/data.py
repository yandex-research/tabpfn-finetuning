import enum
import hashlib
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np
import sklearn.preprocessing
import torch
from loguru import logger
from torch import Tensor

import lib

from .metrics import calculate_metrics as calculate_metrics_
from .util import DataKey, PartKey, PredictionType, Score, TaskType, get_path



_SCORE_SHOULD_BE_MAXIMIZED = {
    Score.ACCURACY: True,
    Score.CROSS_ENTROPY: False,
    Score.MAE: False,
    Score.R2: True,
    Score.RMSE: False,
    Score.ROC_AUC: True,
}


@dataclass(frozen=True)
class Task:
    labels: dict[str, np.ndarray]
    type_: TaskType
    score: Score

    @classmethod
    def from_dir(cls, path: str | Path, split: str = 'default', subsample_power=None) -> 'Task':
        path = get_path(path)
        info = json.loads(path.joinpath('info.json').read_text())
        task_type = TaskType(info['task_type'])
        score = info.get('score')

        if score is None:
            score = {
                TaskType.BINCLASS: Score.ACCURACY,
                TaskType.MULTICLASS: Score.ACCURACY,
                TaskType.REGRESSION: Score.RMSE,
            }[task_type]
        else:
            score = Score(score)
            
        labels = np.load(path / 'Y.npy')
        
        part_idx = {
            part: np.load(path/f'split-{split}/{part}_idx.npy')
            for part in ['train', 'val', 'test']
        }
        
        if subsample_power is not None:
            ds = str(path).split('/')[-1]
            part_idx['train'] = lib.subsamples.SUBSAMPLE_IDXS[f'{ds}-{subsample_power}']

        return Task(
            {
                part: labels[part_idx[part]]
                for part in ['train', 'val', 'test']
            },
            task_type,
            score,
        )

    @classmethod
    def from_dir_old(cls, path: str | Path) -> 'Task':
        path = get_path(path)
        info = json.loads(path.joinpath('info.json').read_text())
        task_type = TaskType(info['task_type'])
        score = info.get('score')
        if score is None and path.name in ('icl', 'ltv-dmdave', 'homesite'):
            score = 'roc-auc'
        if score is None:
            score = {
                TaskType.BINCLASS: Score.ACCURACY,
                TaskType.MULTICLASS: Score.ACCURACY,
                TaskType.REGRESSION: Score.RMSE,
            }[task_type]
        else:
            score = Score(score)
        return Task(
            {
                part: np.load(path / f'Y_{part}.npy')
                for part in ['train', 'val', 'test']
            },
            task_type,
            score,
        )

    def __post_init__(self):
        assert isinstance(self.type_, TaskType)
        assert isinstance(self.score, Score)
        if self.is_regression:
            assert all(
                value.dtype in (np.dtype('float32'), np.dtype('float64'))
                for value in self.labels.values()
            ), 'Regression labels must have dtype=float32'
            for key in self.labels:
                self.labels[key] = self.labels[key].astype('float32')

    @property
    def is_regression(self) -> bool:
        return self.type_ == TaskType.REGRESSION

    @property
    def is_binclass(self) -> bool:
        return self.type_ == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.type_ == TaskType.MULTICLASS

    @property
    def is_classification(self) -> bool:
        return self.is_binclass or self.is_multiclass

    def compute_n_classes(self) -> int:
        assert self.is_binclass or self.is_classification
        return len(np.unique(self.labels['train']))

    def try_compute_n_classes(self) -> None | int:
        return None if self.is_regression else self.compute_n_classes()

    def calculate_metrics(
        self,
        predictions: dict[PartKey, np.ndarray],
        prediction_type: str | PredictionType,
    ) -> dict[PartKey, Any]:
        metrics = {
            part: calculate_metrics_(
                self.labels[part], predictions[part], self.type_, prediction_type
            )
            for part in predictions
        }
        for part_metrics in metrics.values():
            part_metrics['score'] = (
                1.0 if _SCORE_SHOULD_BE_MAXIMIZED[self.score] else -1.0
            ) * part_metrics[self.score.value]
        return metrics  # type: ignore[code]


def load_data(path: str | Path, split: str, features: dict | None, subsample_power: int | None) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    path = get_path(path)
    data = {}
    for key in ['X_num', 'X_bin', 'X_cat', 'X_meta', 'Y']:
        if not path.joinpath(f'{key}.npy').exists():
            print(f'No {key}')
            continue

        arr = np.load(path / f'{key}.npy', allow_pickle=False)
        # Feature selection
        if features is not None and key in features:
            arr = arr[:, features[key]]

        part_idx = {
            part: np.load(path/f'split-{split}/{part}_idx.npy')
            for part in ['train', 'val', 'test']
        }
        
        if subsample_power is not None:
            ds = str(path).split('/')[-1]
            part_idx['train'] = lib.subsamples.SUBSAMPLE_IDXS[f'{ds}-{subsample_power}']
            

        data[key.lower()] = {
            part: arr[part_idx[part]]
            for part in ['train', 'val', 'test']
        }

    return data

def load_data_old(path: str | Path) -> dict[DataKey, dict[PartKey, np.ndarray]]:
    path = get_path(path)
    return {  # type: ignore[code]
        key.lower(): {
            part: np.load(path / f'{key}_{part}.npy', allow_pickle=True)
            for part in ['train', 'val', 'test']
        }
        for key in ['X_num', 'X_bin', 'X_cat', 'Y']
        if path.joinpath(f'{key}_train.npy').exists()
    }


T = TypeVar('T', np.ndarray, Tensor)


@dataclass
class Dataset(Generic[T]):
    """Dataset = Data + Task + simple methods for convenience."""

    data: dict[DataKey, dict[PartKey, T]]
    task: Task

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        split: str = "default",
        features: dict | None = None,
        subsample_power: int | None = None,
    ) -> 'Dataset[np.ndarray]':
        return Dataset(
            load_data(path, split, features, subsample_power),
            Task.from_dir(path, split, subsample_power)
        )

    @classmethod
    def from_dir_old(cls, path: str | Path) -> 'Dataset[np.ndarray]':
        return Dataset(load_data_old(path), Task.from_dir_old(path))

    def __post_init__(self):
        data = self.data

        # >>> Check data types.
        is_numpy = self._is_numpy()
        for key, allowed_dtypes in {
            'x_num': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_bin': [np.dtype('float32')] if is_numpy else [torch.float32],
            'x_cat': [] if is_numpy else [torch.int64],
            'y': (
                [np.dtype('float32'), np.dtype('float64'), np.dtype('int64')]
                if is_numpy
                else [torch.float32, torch.int64]
            ),
        }.items():
            if key in data:
                for part, value in data[key].items():
                    if key == 'x_cat' and is_numpy:
                        assert value.dtype in (
                            np.dtype('int32'),
                            np.dtype('int64'),
                        ) or isinstance(
                            value.dtype, np.dtypes.StrDType  # type: ignore[code]
                        )
                    else:
                        assert value.dtype in allowed_dtypes, (
                            f'The value data["{key}"]["{part}"] has dtype'
                            f' {value.dtype}, but it must be one of {allowed_dtypes}'
                        )

        # >>> Check nans.
        isnan = np.isnan if is_numpy else torch.isnan
        for key in ['x_num', 'x_bin']:
            if key in data:  # type: ignore[code]
                for part, value in data['y'].items():
                    assert not isnan(
                        value  # type: ignore[code]
                    ).any(), f'data["{key}"]["{part}"] contains nans'
        for part, value in data['y'].items():
            assert not isnan(value).any(), f'data["{key}"]["{part}"] contains nans'  # type: ignore[code]

    def _is_numpy(self) -> bool:
        return isinstance(self.data['y']['train'], np.ndarray)

    def __contains__(self, key: DataKey) -> bool:
        return key in self.data

    def __getitem__(self, key: DataKey) -> dict[PartKey, T]:
        return self.data[key]

    def __setitem__(self, key: DataKey, value: dict[PartKey, T]) -> None:
        self.data[key] = value

    @property
    def n_num_features(self) -> int:
        return self.data['x_num']['train'].shape[1] if 'x_num' in self.data else 0

    @property
    def n_bin_features(self) -> int:
        return self.data['x_bin']['train'].shape[1] if 'x_bin' in self.data else 0

    @property
    def n_cat_features(self) -> int:
        return self.data['x_cat']['train'].shape[1] if 'x_cat' in self.data else 0

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_bin_features + self.n_cat_features

    def size(self, part: None | PartKey) -> int:
        return (
            sum(map(len, self.data['y'].values()))
            if part is None
            else len(self.data['y'][part])
        )

    def parts(self) -> Iterable[PartKey]:
        return self.data['y'].keys()

    def compute_cat_cardinalities(self) -> list[int]:
        x_cat = self.data.get('x_cat')
        if x_cat is None:
            return []
        unique = np.unique if self._is_numpy() else torch.unique
        return (
            []
            if x_cat is None
            else [len(unique(column)) for column in x_cat['train'].T]
        )

    def part_data(self, part: PartKey) -> dict[DataKey, T]:
        return {k: self.data[k][part] for k in self.data}

    def to_torch(self, device: None | str | torch.device) -> 'Dataset[Tensor]':
        return Dataset(
            {
                key: {
                    part: torch.as_tensor(value).to(device)
                    for part, value in self.data[key].items()
                }
                for key in self.data
            },
            self.task,
        )


class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    IDENTITY = 'identity'
    NOISY_QUANTILE = 'noisy-quantile'
    NOISY_QUANTILE_OLD = 'noisy-quantile-old'
    ROBUST_SMOOTH_CLIP = 'robust-smooth-clip'


class RobustScaleSmoothClipTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        # don't deal with dataframes for simplicity
        assert isinstance(X, np.ndarray)
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        max = np.max(X, axis=-2)
        min = np.min(X, axis=-2)
        idxs = quant_diff == 0.0
        # on indexes where the quantile difference is zero, do min-max scaling instead
        quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        # if feature is constant on the training data,
        # set factor to zero so that it is also constant at prediction time
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)


# Inspired by: https://github.com/Yura52/rtdl/blob/a4c93a32b334ef55d2a0559a4407c8306ffeeaee/lib/data.py#L20
def transform_num(
    X_num: dict[PartKey, np.ndarray], policy: str | NumPolicy, seed: None | int, fillna: bool = True
) -> dict[PartKey, np.ndarray]:
    policy = NumPolicy(policy)
    X_num_train = X_num['train']
    if policy == NumPolicy.STANDARD:
        normalizer = sklearn.preprocessing.StandardScaler()
    elif policy == NumPolicy.NOISY_QUANTILE:
        normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
            output_distribution='normal',
            subsample=1_000_000_000,
            random_state=seed,
        )
        assert seed is not None
        X_num_train = X_num_train + np.random.RandomState(seed).normal(
            0.0, 1e-5, X_num_train.shape
        ).astype(X_num_train.dtype)
    elif policy == NumPolicy.NOISY_QUANTILE_OLD:
        normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=max(min(X_num['train'].shape[0] // 30, 1000), 10),
            output_distribution='normal',
            subsample=1_000_000_000,
            random_state=seed,
        )
        assert seed is not None
        stds = np.std(X_num_train, axis=0, keepdims=True)
        noise_std = 1e-3 / np.maximum(stds, 1e-3)  # type: ignore[code]
        X_num_train = X_num_train + noise_std * np.random.default_rng(
            seed
        ).standard_normal(X_num_train.shape)
    elif policy == NumPolicy.IDENTITY:
        normalizer = sklearn.preprocessing.FunctionTransformer()
    elif policy == NumPolicy.ROBUST_SMOOTH_CLIP:
        normalizer = RobustScaleSmoothClipTransform()
    else:
        raise ValueError(f'Unknown policy={policy}')

    normalizer.fit(X_num_train)
    # return {k: normalizer.transform(v) for k, v in X_num.items()}  # type: ignore[code]
    if fillna:
        return {
            # NOTE: this is not a good way to process NaNs.
            # This is just a quick hack to stop failing on some datasets.
            # NaNs are replaced with zeros (zero is the mean value for all features after
            # the conventional preprocessing techniques.
            k: np.nan_to_num(normalizer.transform(v)).astype(np.float32)  # type: ignore[code]
            for k, v in X_num.items()
        }
    else:
        return {
            k: normalizer.transform(v).astype(np.float32)  # type: ignore[code]
            for k, v in X_num.items()
        }


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


def transform_cat(
    X_cat: dict[PartKey, np.ndarray], policy: str | CatPolicy
) -> dict[PartKey, np.ndarray]:
    policy = CatPolicy(policy)

    # The first step is always the ordinal encoding,
    # even for the one-hot encoding.
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',  # type: ignore[code]
        unknown_value=unknown_value,  # type: ignore[code]
        dtype='int64',  # type: ignore[code]
    ).fit(X_cat['train'])
    X_cat = {k: encoder.transform(v) for k, v in X_cat.items()}
    max_values = X_cat['train'].max(axis=0)
    for part in ['val', 'test']:
        part = cast(PartKey, part)
        for column_idx in range(X_cat[part].shape[1]):
            X_cat[part][X_cat[part][:, column_idx] == unknown_value, column_idx] = (
                max_values[column_idx] + 1
            )

    # >>> encode
    if policy == CatPolicy.ORDINAL:
        return X_cat
    elif policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore', sparse_output=False, dtype=np.float32  # type: ignore[code]
        )
        encoder.fit(X_cat['train'])
        return {k: cast(np.ndarray, encoder.transform(v)) for k, v in X_cat.items()}
    else:
        raise ValueError(f'Unknown policy={policy}')


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


def standardize_labels(
    y: dict[PartKey, np.ndarray]
) -> tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    y = {k: v.astype(np.float32) for k,v in y.items()}
    mean = float(y['train'].mean())
    std = float(y['train'].std()) + 1e-20
    return {k: (v - mean) / std for k, v in y.items()}, RegressionLabelStats(
        mean=mean, std=std
    )


def build_dataset(
    *,
    path: str | Path,
    num_policy: None | str | NumPolicy = None,
    cat_policy: None | str | CatPolicy = None,
    seed: int = 0,
    split: str = "default",
    cache: bool = False,
    old: bool = False,
    fillna: bool = True,
    feature_selection: bool = False,
    subsample_power: int | None = None,
) -> Dataset[np.ndarray]:
    path = get_path(path)
    if cache:
        args = locals()
        args.pop('cache')
        args.pop('path')
        cache_path = lib.CACHE_DIR / (
            f'build_dataset__{path.name}__{hashlib.md5(str(args).encode("utf-8")).hexdigest()}.pickle'
        )
        if cache_path.exists():
            cached_args, cached_value = pickle.loads(cache_path.read_bytes())
            assert args == cached_args, f'Hash collision for {cache_path}'
            logger.info(f"Using cached dataset: {cache_path.name}")
            return cached_value
    else:
        args = None
        cache_path = None

    features = None
    dataset = Dataset.from_dir(path, split, features, subsample_power) if not old else Dataset.from_dir_old(path)
    if num_policy is not None:
        dataset['x_num'] = transform_num(dataset['x_num'], num_policy, seed, fillna)
    if cat_policy is not None:
        dataset['x_cat'] = transform_cat(dataset['x_cat'], cat_policy)

    if cache_path is not None:
        cache_path.write_bytes(pickle.dumps((args, dataset)))
    return dataset
