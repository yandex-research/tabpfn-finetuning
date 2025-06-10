# Finetune TabPFNv2 model

# > lifehack 2.0: Debug on error
# import sys
# import pudb
# def exception_handler(exc_type, exc_value, exc_traceback):
#     print("Uncaught exception:", exc_type, exc_value)
#     pudb.post_mortem(tb=exc_traceback)
# sys.excepthook = exception_handler

import os
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import delu
import numpy as np
import rtdl_num_embeddings
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from typing_extensions import Callable, NotRequired, TypedDict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    PowerTransformer,
    FunctionTransformer,
)

import lib
from lib import KWArgs, PartKey
from lib.tabpfn.model.layer import PerFeatureEncoderLayer
import lib.tabpfn.lora_utils as lora_utils
from lib.tabpfn.utils import _transform_borders_one, translate_probs_across_borders

# regression losses
from lib.tabpfn.model.bar_distribution import (
    FullSupportBarDistribution,
)

EvalOut = tuple[dict[PartKey, Any], dict[PartKey, np.ndarray], int]


def regression_output_transform(
    target_transform,
    criterion,
    renormalized_criterion,
    softmax_temperature: float = 0.9,
):
    "transform model outputs into original values for regression; extracted from regressor.py in tabpfn"

    std_borders = criterion.borders.cpu().numpy()
    logit_cancel_mask, descending_borders, borders_t = _transform_borders_one(
        std_borders,
        target_transform=target_transform,
        repair_nan_borders_after_transform=True,
    )
    if descending_borders:
        borders_t = borders_t.flip(-1)  # type: ignore

    device = lib.get_device()

    def transform(out):
        logits = translate_probs_across_borders(
            out.float() / softmax_temperature,
            frm=torch.as_tensor(borders_t, device=device),
            to=criterion.borders.to(device),
        )
        if logit_cancel_mask is not None:
            out = out.clone()  # pyright: ignore
            out[..., logit_cancel_mask] = float("-inf")

        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()
        logits = logits.cpu()

        return renormalized_criterion.mean(logits)

    return transform


class LayerStack(nn.Module):
    """Similar to nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert self.min_num_layers_layer_dropout == self.num_layers, (
                "half_layers only works without layer dropout"
            )
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:  # pyright: ignore
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)  # type: ignore
            else:
                x = layer(x, **kwargs)

        return x


class TabPFN2(nn.Module):
    "TabPFN2 with learnable feature embeddings"

    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: int,
        tabpfn_config: dict,
        is_regression: bool,
        bins: None | list[Tensor],
        num_embeddings: None | dict = None,
        affine_layer_norm: bool = False,
        untie_value_embeddings: bool = False,
        untie_pos_embeddings: bool = False,
        random_init: bool = False,
    ) -> None:
        print(untie_value_embeddings)
        assert n_num_features or n_bin_features or cat_cardinalities
        super().__init__()

        self.is_regression = is_regression
        checkpoint = (
            "tabpfn-v2-regressor.ckpt" if is_regression else "tabpfn-v2-classifier.ckpt"
        )

        state_dict = torch.load(
            lib.PROJECT_DIR / checkpoint,
            weights_only=True,
        )["state_dict"]

        # Loading just the relevant part of the state dict
        extract_state_dict = lambda pref: (
            {
                k.removeprefix(pref): v
                for k, v in state_dict.items()
                if k.startswith(pref)
            }
        )

        self.cat_cardinalities = cat_cardinalities

        # >>> Feature Embeddings (and target embeddings)
        #
        #  We start simple with Linear emebddings without and unshare their parameters,
        #  and also init positional embeddings a slightly different

        # initialized linear layer for the first feature

        assert num_embeddings is not None

        if bins is not None:
            num_embeddings["bins"] = bins
        else:
            num_embeddings["n_features"] = n_num_features

        # positional embeddings init from random subspace
        pos_embs = nn.Linear(48, 192)
        pos_embs.load_state_dict(
            extract_state_dict("feature_positional_embedding_embeddings.")
        )

        if not untie_pos_embeddings:
            self.pos_embs = pos_embs

        self.untie_value_embeddings = untie_value_embeddings
        self.untie_pos_embeddings = untie_pos_embeddings

        if n_num_features > 0 and untie_value_embeddings:
            # Saving just the init, all embeddings are united
            self.m_num = lib.deep.make_module(
                **num_embeddings,
                d_embedding=tabpfn_config["emsize"],
                bias=untie_pos_embeddings,
            )
            weight_init = (
                state_dict["encoder.5.layer.weight"][:, 0]
                .unsqueeze(0)
                .repeat(n_num_features, 1)
            )
            if untie_pos_embeddings:
                bias_init = pos_embs(
                    torch.randn(n_num_features, tabpfn_config["emsize"] // 4)
                )
                self.pos_embs = None
            else:
                # just like tabpfn inference (random subspace with add before transformer)
                self.pos_embs = pos_embs

            if isinstance(self.m_num, lib.deep.LinearEmbeddings):
                self.m_num.weight.data = weight_init
                if untie_pos_embeddings:
                    self.m_num.bias.data = bias_init  # pyright: ignore
            elif isinstance(
                self.m_num,
                (lib.deep.PiecewiseLinearEmbeddings, lib.deep.PeriodicEmbeddingsV2),
            ):
                self.m_num.linear0.weight.data = weight_init  # pyright: ignore
                if untie_pos_embeddings:
                    self.m_num.linear0.bias.data = bias_init  # pyright: ignore
        elif (
            n_num_features > 0
            and not untie_value_embeddings
            and not untie_pos_embeddings
        ):
            # This is TabPFN 101 essentially
            self.m_num = nn.Linear(1, tabpfn_config["emsize"], bias=False)
            weight_init = state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(1)
            self.m_num.weight.data = weight_init

        elif n_num_features > 0 and not untie_value_embeddings and untie_pos_embeddings:
            # This is weird. value embeds are shared, positional emebds are not. TODO maybe support this
            raise ValueError("This is not supported yet (untie val + no untie pos)")
        else:
            self.m_num = None

        if cat_cardinalities and untie_value_embeddings:
            self.m_cat = lib.deep.CategoricalEmbeddings1d(
                cat_cardinalities, tabpfn_config["emsize"]
            )
            # TODO. Different strategies for positional embedding inits?
            # For now just vanilla per feature embeddings
            if untie_pos_embeddings:
                cat_pos_embs = pos_embs(
                    torch.randn(len(cat_cardinalities), tabpfn_config["emsize"] // 4)
                )

            for i, c in enumerate(cat_cardinalities):
                embs = (
                    # kind-of positional embedding (or zero)
                    (
                        cat_pos_embs[i, :].unsqueeze(0)
                        if untie_pos_embeddings  # pyright: ignore
                        else torch.zeros(1, tabpfn_config["emsize"])
                    )
                    # kind-of value embedding
                    + (
                        state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(0)
                        *
                        # c+1 is a hack for unk features. c-1 is for min-max like scaling
                        torch.arange(c + 1).unsqueeze(1)
                        / (c - 1)
                    )
                )
                self.m_cat.embeddings[i].weight.data = embs
        elif (
            cat_cardinalities
            and not untie_value_embeddings
            and not untie_pos_embeddings
        ):
            # This will work fine, I have tests
            if n_num_features > 0:
                self.m_cat = self.m_num
            else:
                # This is just Mercedes
                self.m_cat = nn.Linear(1, tabpfn_config["emsize"], bias=False)
                weight_init = state_dict["encoder.5.layer.weight"][:, 0].unsqueeze(1)
                self.m_cat.weight.data = weight_init
        elif cat_cardinalities and not untie_value_embeddings and untie_pos_embeddings:
            # This is weird. value embeds are shared, positional emebds are not. TODO maybe support this
            raise ValueError("This is not supported yet (untie val + no untie pos)")
        else:
            self.m_cat = None

        # embeddings for the target variable
        # first coord is the target variable itself, second is nan-indicator
        # bias is always applied

        layer_key = "1" if is_regression else "2"
        self.y_embedding_weight = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 0]
        )
        self.y_embedding_nan_ind = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.weight"][:, 1]
        )
        self.y_embedding_bias = nn.Parameter(
            state_dict[f"y_encoder.{layer_key}.layer.bias"]
        )

        # >>> TabPFN per-feature transformer (loading the model from checkpoint)
        ninp = tabpfn_config["emsize"]
        nhead = tabpfn_config["nhead"]
        nhid = tabpfn_config["emsize"] * tabpfn_config["nhid_factor"]
        nlayers = tabpfn_config["nlayers"]

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation="gelu",
            zero_init=False,
            precomputed_kv=None,
            multiquery_item_attention_for_test_set=True,
            layer_norm_with_elementwise_affine=affine_layer_norm,
        )
        nlayers_encoder = nlayers

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=False,  # TODO this may be needed
            min_num_layers_layer_dropout=None,
        )
        self.transformer_encoder.load_state_dict(
            extract_state_dict("transformer_encoder."), strict=not affine_layer_norm
        )

        self.decoder = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, n_classes),  # TODO handle this outside for regression
        )
        self.decoder.load_state_dict(extract_state_dict("decoder_dict.standard."))

    def forward(
        self,
        *,
        x_num: None | Tensor = None,
        x_cat: None | Tensor = None,
        y_train: Tensor,
    ) -> Tensor:
        bs = y_train.shape[0]
        train_size = y_train.shape[1]

        # >>> Input feature embeddings
        x = []
        if x_num is not None:
            if isinstance(
                self.m_num,
                (lib.deep.PiecewiseLinearEmbeddings, lib.deep.PeriodicEmbeddingsV2),
            ):
                seq_len = x_num.shape[1]
                n_features = x_num.shape[2]
                x.append(
                    x_num
                    if self.m_num is None
                    else self.m_num(x_num.flatten(0, 1)).view(
                        bs, seq_len, n_features, -1
                    )
                )
            else:
                if not self.untie_value_embeddings:
                    x.append(
                        x_num if self.m_num is None else self.m_num(x_num.unsqueeze(-1))
                    )
                else:
                    x.append(x_num if self.m_num is None else self.m_num(x_num))

        if x_cat is None:
            assert self.m_cat is None
        else:
            assert self.m_cat is not None
            if not self.untie_value_embeddings:
                # min-max scale if using linear layer after this
                x_cat_max = x_cat[:, :train_size].max(dim=1, keepdim=True).values
                x_cat = (x_cat / x_cat_max).unsqueeze(-1)

            x_cat = self.m_cat(x_cat)
            x.append(x_cat)

        x_inp = torch.cat(x, dim=2)
        total_size = x_inp.shape[1]

        # >>> Target embeddings

        # placeholder from train set
        y_mult = y_train.mean(dim=1, keepdim=True)
        if not self.is_regression:
            y_mult = torch.round(y_mult)
        y_test = x_inp.new_ones(bs, total_size - train_size) * y_mult
        nan_ind = x_inp.new_zeros(bs, total_size)
        nan_ind[:, train_size:] = -2.0
        y_emb = (
            torch.cat([y_train, y_test], dim=1).view(bs, -1, 1, 1).float()
            * self.y_embedding_weight.view(1, 1, 1, -1).repeat(bs, 1, 1, 1)
            + nan_ind.view(bs, -1, 1, 1)
            * self.y_embedding_nan_ind.view(1, 1, 1, -1).repeat(bs, 1, 1, 1)
            + self.y_embedding_bias.view(1, 1, 1, -1).repeat(bs, 1, 1, 1)
        )

        # Subspace positional embeddings from TabPFN
        if self.pos_embs is not None:
            bs, _, n_features, d_emb = x_inp.shape
            x_inp = (
                x_inp
                + self.pos_embs(
                    torch.randn(n_features, d_emb // 4, device=x_inp.device)
                )[None, None]
            )

        # >>> Transformer forward pass and prediction
        x_inp = torch.cat([x_inp, y_emb], dim=2)  # bs, s, f+1, #emsize
        encoder_out = self.transformer_encoder(
            x_inp,
            half_layers=False,
            cache_trainset_representation=False,
            single_eval_pos=train_size,
        )

        return self.decoder(encoder_out[:, train_size:, -1])


class Config(TypedDict):
    seed: int
    data: KWArgs
    bins: NotRequired[KWArgs]
    model: KWArgs
    optimizer: KWArgs
    epoch_size: int
    batch_size: int
    # seq_len_train: int
    seq_len_pred: int

    patience: int
    n_epochs: int
    amp: NotRequired[bool]  # Automatic mixed precision in bfloat16.
    n_lr_warmup_epochs: NotRequired[int]
    gradient_clipping_norm: NotRequired[float]
    parameter_statistics: NotRequired[bool]
    full_finetune: NotRequired[bool]
    finetune_mode: NotRequired[
        Literal[
            "full",
            "ln",
            "head",
            "ln+head",
            "embeds",
            "embeds+head",
            "embeds+ln+head",
            "top_n_layers",
            "lora",
        ]
    ]
    finetune_layers: NotRequired[int]
    lora_config: NotRequired[lora_utils.LoraConfig]
    randperm: NotRequired[bool]


class CandidateQueue:
    def __init__(
        self, train_size: int, n_candidates: int | float, device: torch.device
    ) -> None:
        assert train_size > 0
        if isinstance(n_candidates, int):
            assert 0 < n_candidates < train_size
            self._n_candidates = n_candidates
        else:
            assert 0.0 < n_candidates < 1.0
            self._n_candidates = int(n_candidates * train_size)
        self._train_size = train_size
        self._candidate_queue = torch.tensor([], dtype=torch.int64, device=device)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._candidate_queue) < self._n_candidates:
            self._candidate_queue = torch.cat(
                [
                    self._candidate_queue,
                    torch.randperm(
                        self._train_size, device=self._candidate_queue.device
                    ),
                ]
            )
        candidate_indices, self._candidate_queue = self._candidate_queue.split(
            [self._n_candidates, len(self._candidate_queue) - self._n_candidates]
        )
        return candidate_indices


def main(
    config: Config, output: str | Path, *, force: bool = False
) -> None | lib.JSONDict:
    # >>> start
    config, output = lib.check(config, output, config_type=Config)

    if not lib.start(output, force=force):
        return None

    lib.show_config(config)  # type: ignore[code]
    output = Path(output)
    delu.random.seed(config["seed"])
    device = lib.get_device()
    report = lib.create_report(config)  # type: ignore[code]

    # >>> dataset
    dataset = lib.data.build_dataset(**config["data"])
    if dataset.task.is_regression:
        dataset.data["y"], regression_label_stats = lib.data.standardize_labels(
            dataset.data["y"]
        )

        target_transform = config["model"].pop("target_transform", None)

        if target_transform == "power":
            target_transform = Pipeline(
                [("power", PowerTransformer()), ("standard", StandardScaler())]
            ).fit(dataset.data["y"]["train"].reshape(-1, 1))
        elif target_transform == "quantile":
            target_transform = QuantileTransformer(
                output_distribution="normal", random_state=config["seed"]
            ).fit(dataset.data["y"]["train"].reshape(-1, 1))
        elif target_transform is None:
            target_transform = FunctionTransformer(func=None)
        else:
            raise ValueError(f"Unknown target_transform {target_transform}")

        dataset.data["y"] = {
            part: target_transform.transform(dataset.data["y"][part].reshape(-1, 1))
            .astype(np.float32)  # pyright: ignore
            .squeeze()
            for part in dataset.data["y"]
        }
    else:
        regression_label_stats = None

    # Convert binary features to categorical features.
    if dataset.n_bin_features > 0:
        x_bin = dataset.data.pop("x_bin")
        # Remove binary features with just one unique value in the training set.
        # This must be done, otherwise, the script will fail on one specific dataset
        # from the "why" benchmark.
        n_bin_features = x_bin["train"].shape[1]
        good_bin_idx = [
            i for i in range(n_bin_features) if len(np.unique(x_bin["train"][:, i])) > 1
        ]
        if len(good_bin_idx) < n_bin_features:
            x_bin = {k: v[:, good_bin_idx] for k, v in x_bin.items()}

        if dataset.n_cat_features == 0:
            dataset.data["x_cat"] = {
                part: np.zeros((dataset.size(part), 0), dtype=np.int64)
                for part in x_bin
            }
        for part in x_bin:
            dataset.data["x_cat"][part] = np.column_stack(
                [dataset.data["x_cat"][part], x_bin[part].astype(np.int64)]
            )
        del x_bin

    dataset = dataset.to_torch(device)
    Y_train = dataset.data["y"]["train"].to(dtype=torch.float32, device=device)

    # >>> model
    if "bins" in config:
        compute_bins_kwargs = (
            {
                "y": Y_train.to(
                    torch.long if dataset.task.is_classification else torch.float
                ),
                "regression": dataset.task.is_regression,
                "verbose": True,
            }
            if "tree_kwargs" in config["bins"]
            else {}
        )
        bin_edges = rtdl_num_embeddings.compute_bins(
            dataset["x_num"]["train"], **config["bins"], **compute_bins_kwargs
        )
        logger.info(f"Bin counts: {[len(x) - 1 for x in bin_edges]}")
    else:
        bin_edges = None

    logger.info(
        f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    n_classes = 10 if dataset.task.is_classification else 5_000

    model = TabPFN2(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.compute_cat_cardinalities(),
        n_classes=n_classes,
        is_regression=dataset.task.is_regression,
        affine_layer_norm="ln" in config.get("finetune_mode", ""),
        **config["model"],
        bins=bin_edges,
    ).to(device)

    report["n_parameters"] = lib.deep.get_n_parameters(model)
    logger.info(f"n_parameters = {report['n_parameters']}")
    report["prediction_type"] = "labels" if dataset.task.is_regression else "probs"
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # >>> training
    all_param_names = [name for name, _ in model.named_parameters()]
    param_names = {
        "ln": [name for name in all_param_names if "layer_norms" in name],
        "head": [name for name in all_param_names if "decoder.2" in name],
        "embeds": [
            name
            for name in all_param_names
            if "m_num" in name
            or "m_cat" in name
            or "y_embedding" in name
            or "pos_embs" in name
        ],
    }

    if (
        config.get("full_finetune", False)
        or config.get("finetune_mode", None) == "full"
    ):
        params = lib.deep.make_parameter_groups(model)
    elif config.get("finetune_mode", "") == "":  # ?
        params = [
            p
            for pn, p in model.named_parameters()
            if "m_num" in pn or "m_cat" in pn or "y_embedding" in pn
        ]
    elif config.get("finetune_mode") == "top_n_layers":
        finetune_layers = config.get("finetune_layers", None)
        assert finetune_layers is not None, "Specify how many top layers to finetune"
        params = [
            p for p in model.transformer_encoder.layers[-finetune_layers:].parameters()
        ] + [p for p in model.decoder.parameters()]
    elif config.get("finetune_mode") == "full+ln":
        params = lib.deep.make_parameter_groups(model)
    elif config.get("finetune_mode", "") == "lora":
        assert config.get("lora_config", None) is not None
        lora_utils.replace_with_loras(model, config["lora_config"])  # type: ignore
        params = []
        for pn, p in model.named_parameters():
            if "lora" in pn:
                params.append(p)
            else:
                p.requires_grad = False
    else:
        finetuning_components = config.get("finetune_mode", "").split("+")
        param_names_finetune = [
            pname for k in finetuning_components for pname in param_names[k]
        ]
        params = []
        for pn, p in model.named_parameters():
            if pn in param_names_finetune:
                params.append(p)
            else:
                p.requires_grad = False

    if config.get("finetune_mode", None) != "full":
        print("Num trainable: ", sum(p.numel() for p in params))  # type: ignore

    optimizer = lib.deep.make_optimizer(
        **config["optimizer"],
        params=params,
    )
    if dataset.task.is_regression:
        assert regression_label_stats is not None
        borders = torch.load(
            lib.PROJECT_DIR / "tabpfn-v2-regressor.ckpt",
            weights_only=True,
        )["state_dict"]["criterion.borders"].to(device)
        loss_fn = FullSupportBarDistribution(borders)
        renormalized_criterion = FullSupportBarDistribution(
            loss_fn.borders * regression_label_stats.std + regression_label_stats.mean,
        ).float()
        pred_transform = regression_output_transform(
            target_transform,  # pyright: ignore
            loss_fn,
            renormalized_criterion,
            softmax_temperature=config["model"]["tabpfn_config"].get(
                "softmax_temperature", 0.9
            ),
        )
    else:
        loss_fn = F.cross_entropy
    gradient_clipping_norm = config.get("gradient_clipping_norm")

    epoch_size = config["epoch_size"]
    eval_batch_size = 4096
    chunk_size = None
    generator = torch.Generator(device).manual_seed(config["seed"])

    report["metrics"] = {"val": {"score": -math.inf}}
    if "n_lr_warmup_epochs" in config:
        n_warmup_steps = min(10000, config["n_lr_warmup_epochs"] * epoch_size)
        n_warmup_steps = max(1, math.trunc(n_warmup_steps / epoch_size)) * epoch_size
        logger.info(f"{n_warmup_steps=}")
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=n_warmup_steps
        )
    else:
        lr_scheduler = None

    timer = delu.tools.Timer()
    parameter_statistics = config.get("parameter_statistics", config["seed"] == 1)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    amp_enabled = (
        config.get("amp", False)
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    logger.info(f"AMP enabled: {amp_enabled}")

    @torch.autocast(
        device.type, enabled=amp_enabled, dtype=torch.bfloat16 if amp_enabled else None
    )
    def apply_model(part: PartKey, idx_train: Tensor, idx: Tensor) -> Tensor:
        if "x_num" in dataset.data:
            x_num = torch.cat(
                [
                    dataset.data["x_num"]["train"][idx_train],
                    dataset.data["x_num"][part][idx],
                ],
                dim=1,
            )
        else:
            x_num = None

        if "x_cat" in dataset.data:
            x_cat = torch.cat(
                [
                    dataset.data["x_cat"]["train"][idx_train],
                    dataset.data["x_cat"][part][idx],
                ],
                dim=1,
            )
        else:
            x_cat = None

        # For flash attention reproducibility
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            out = model.forward(
                x_num=x_num, x_cat=x_cat, y_train=Y_train[idx_train]
            ).float()

        if dataset.task.is_classification:
            return out[..., : dataset.task.compute_n_classes()]
        else:
            return out

    @torch.inference_mode()
    def evaluate(parts: list[PartKey], eval_batch_size: int) -> EvalOut:
        model.eval()
        predictions: dict[PartKey, np.ndarray] = {}
        # using the whole train on evaluation
        idx_train = torch.arange(dataset.size("train"), device=device)

        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(
                                    part, idx_train.unsqueeze(0), idx.unsqueeze(0)
                                ).squeeze(0)
                                for idx in tqdm(
                                    torch.arange(
                                        dataset.size(part),
                                        device=device,
                                    ).split(eval_batch_size)
                                )
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    delu.cuda.free_memory()
                    logger.warning(f"eval_batch_size = {eval_batch_size}")
                else:
                    break
            if not eval_batch_size:
                RuntimeError("Not enough memory even for eval_batch_size=1")

        if regression_label_stats is not None:
            predictions = {
                k: pred_transform(torch.from_numpy(v).to(device)).cpu().numpy()  # pyright: ignore
                for k, v in predictions.items()
            }
        else:
            predictions = {
                k: scipy.special.softmax(v, axis=-1) for k, v in predictions.items()
            }
            if dataset.task.is_binclass:
                predictions = {k: v[..., 1] for k, v in predictions.items()}

        metrics = (
            dataset.task.calculate_metrics(predictions, report["prediction_type"])
            if lib.are_valid_predictions(predictions)
            else {x: {"score": -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def train_loop(
        *,
        step_fn: Callable[[Tensor, Tensor], Tensor],
        eval_fn: Callable[..., tuple],
        n_steps: int,
        patience: int,
        report_key: str,
        chunk_size=None,
        eval_batch_size=eval_batch_size,
    ):
        def save_checkpoint(step) -> None:
            lib.dump_checkpoint(
                output,
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.get_state(),
                    "random_state": delu.random.get_state(),
                    "early_stopping": early_stopping,
                    "report": report,
                    "timer": timer,
                    "training_log": training_log,
                }
                | (
                    {}
                    if lr_scheduler is None
                    else {"lr_scheduler": lr_scheduler.state_dict()}
                ),
            )
            lib.dump_report(output, report)
            lib.backup_output(output)

        step = 0
        early_stopping = delu.tools.EarlyStopping(patience, mode="max")
        report[report_key] = {"metrics": {"val": {"score": -math.inf}}}

        while n_steps == -1 or step < n_steps:
            print(f"[...] {output} | {timer}")

            # >>>
            model.train()
            epoch_losses = []
            logs_train = defaultdict(list)

            idx_queue = CandidateQueue(
                dataset.size("train"),
                n_candidates=config["batch_size"] * config["seq_len_pred"],
                device=device,
            )
            delu.cuda.free_memory()

            for _ in tqdm(
                range(epoch_size),
                desc=f"Epoch {step // epoch_size} Step {step}",
            ):
                if config.get("randperm", False):
                    idx = torch.randperm(dataset.size("train"), device=device)
                    idx = idx[: config["batch_size"] * config["seq_len_pred"]]
                else:
                    idx = next(idx_queue)
                idx = idx.view(config["batch_size"], -1)

                mask = idx.new_ones(
                    (config["batch_size"], dataset.size("train")), dtype=torch.bool
                )
                mask[
                    torch.arange(config["batch_size"], device=mask.device).unsqueeze(
                        -1
                    ),
                    idx,
                ] = False
                idx_train = (
                    torch.arange(dataset.size("train"), device=idx.device)
                    .expand(config["batch_size"], dataset.size("train"))[mask]
                    .view(config["batch_size"], -1)
                )

                optimizer.zero_grad()
                loss = step_fn(idx_train, idx)
                loss.backward()

                for k, v in log_dict.items():
                    logs_train[k].append(v)

                if parameter_statistics and (
                    step % epoch_size == 0  # The first batch of the epoch.
                    or step // epoch_size == 0  # The first epoch.
                ):
                    for k, v in lib.deep.compute_parameter_stats(model).items():
                        writer.add_scalars(
                            f"{report_key}/{k}", v, step, timer.elapsed()
                        )
                        del k, v

                if gradient_clipping_norm is not None:
                    nn.utils.clip_grad.clip_grad_norm_(
                        model.parameters(), gradient_clipping_norm
                    )
                optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                step += 1
                epoch_losses.append(loss.detach())

            epoch_losses = torch.stack(epoch_losses).tolist()
            mean_loss = statistics.mean(epoch_losses)

            metrics, predictions, eval_batch_size = eval_fn(["val"], eval_batch_size)
            metrics["train"] = {}
            for k, v in logs_train.items():
                metrics["train"][k] = np.mean(v).item()

            training_log.append(
                {
                    "epoch-losses": epoch_losses,
                    "metrics": metrics,
                    "time": timer.elapsed(),
                }
            )
            lib.print_metrics(mean_loss, metrics)
            writer.add_scalars(
                f"{report_key}/loss", {"train": mean_loss}, step, timer.elapsed()
            )
            for part in metrics:
                for k in metrics[part].keys():
                    if k != "score":
                        continue
                    writer.add_scalars(
                        f"{report_key}/{k}",
                        {part: metrics[part][k]},
                        step,
                        timer.elapsed(),
                    )

            if metrics["val"]["score"] > report[report_key]["metrics"]["val"]["score"]:
                print("🌸 New best epoch! 🌸")
                report[report_key]["best_step"] = step
                report[report_key]["metrics"] = metrics
                save_checkpoint(step)
                lib.dump_predictions(output, predictions)

            early_stopping.update(metrics["val"]["score"])
            if early_stopping.should_stop() or not lib.are_valid_predictions(
                predictions
            ):
                break

            print()
        return chunk_size, eval_batch_size

    def step_fn(idx_train, idx):
        "idx is big set of datasets"
        if dataset.task.is_classification:
            return loss_fn(
                apply_model("train", idx_train, idx).permute(0, 2, 1),
                Y_train[idx].long(),
            )
        else:
            return loss_fn(
                apply_model("train", idx_train, idx).permute(1, 0, 2),
                Y_train[idx].transpose(0, 1),
            ).mean()

    logger.info("Score before training")

    report["zero-shot-metrics"], _, _ = evaluate(["val", "test"], eval_batch_size)
    lib.print_metrics(0, report["zero-shot-metrics"])

    log_dict = {}
    timer.run()
    chunk_size, eval_batch_size = train_loop(
        step_fn=step_fn,
        eval_fn=evaluate,
        n_steps=config["n_epochs"],
        patience=config["patience"],
        report_key="train",
        chunk_size=chunk_size,
    )
    report["time"] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)["model"])
    logger.info("Final Eval")
    report["metrics"], predictions, _ = evaluate(
        ["train", "val", "test"], eval_batch_size
    )
    report["chunk_size"] = chunk_size
    report["eval_batch_size"] = eval_batch_size
    lib.dump_predictions(output, predictions)
    lib.dump_summary(output, lib.summarize(report))
    # to free-up space
    if config["seed"] > 0:
        os.remove(output / "checkpoint.pt")
    lib.finish(output, report)
    return report


if __name__ == "__main__":
    lib.configure_libraries()
    lib.run_MainFunction_cli(main)
