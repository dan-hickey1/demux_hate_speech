# from abc import ABC, abstractmethod
from typing import List, Dict, Iterable, Any, Optional, Union, Tuple
from copy import deepcopy
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, f1_score

from emorec.trainer import BaseTrainer
from emorec.train_utils import Correlations, MultilabelConditionalWeights, EarlyStopping
from emorec.utils import flatten_list
import math


import logging
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup, AdamW

from emorec.logging_utils import ExperimentHandler

def result_str(results: Dict[str, float]):
    return ", ".join(
        [
            f"{key}={value:.4f}"
            if isinstance(value, float)
            else f"{key}={value}"
            for key, value in results.items()
        ]
    )

class MultilabelEmotionTrainer(BaseTrainer):
    """Base trainer class for Multilabel Emotion Recognition.

    Assumption here is that the outputs of the dataset
    conform to `transformers` conventions, i.e. the first output
    is expected to be a dictionary.

    Attributes:
        Check `BaseTrainer`.
    """

    early_stopping_metric = "jaccard_score"

    def train_end(self):
        self.exp_handler.log()
        self._save_best_model()
        self.exp_handler.aggregate_results()
        self.exp_handler.plot(
            groups=[
                [
                    f"{emotion_cls}_f1"
                    if isinstance(emotion_cls, str)
                    else "-".join(emotion_cls) + "_f1"
                    for emotion_cls in self.dev_dataset.emotions
                ]
            ]
            if self.do_eval
            else None
        )

    def batch_labels(self, batch: Iterable[Any]):
        return batch[1]

    def batch_len(self, batch):
        """Returns length of tensors inside inputs dict."""
        return len(next(iter(batch[0].values())))

    def input_batch_kwargs(self, batch):
        """Returns input dict."""
        return batch[0]

    def calculate_cls_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, train: bool
    ) -> torch.Tensor:
        """Calculates the average BCE loss of all emotions across the batch."""
        criterion = nn.BCEWithLogitsLoss()
        return criterion(logits, labels)

    def evaluation_metrics(
        self,
        eval_true: List[int],
        eval_preds: List[int],
        eval_probs: List[float],
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """Computes evaluation F1s and JS.

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.
            data_loader: DataLoader where data came from.

        Returns:
            A dict of metrics.
        """
        macro_f1 = f1_score(
            eval_true, eval_preds, average="macro", zero_division=1
        )
        micro_f1 = f1_score(
            eval_true, eval_preds, average="micro", zero_division=1
        )

        js = jaccard_score(
            eval_true, eval_preds, average="samples", zero_division=1
        )

        f1_scores = f1_score(
            eval_true, eval_preds, average=None, zero_division=1
        )

        roc_auc_scores = roc_auc_score(eval_true, eval_probs, average=None)

        max_f1_scores = []
        max_f1_thresholds = []
        
        for true, probs in zip(np.array(eval_true).T, np.array(eval_probs).T):
            max_f1 = 0
            max_threshold = 0
            for threshold in np.arange(0, 1, 0.05):
                preds = (probs > threshold).astype('int')
                f1 = f1_score(true, preds)
                if f1 > max_f1:
                    max_f1 = f1
                    max_threshold = threshold
            
            max_f1_scores.append(max_f1)
            max_f1_thresholds.append(max_threshold)

        results = dict(jaccard_score=js, micro_f1=micro_f1, macro_f1=macro_f1)
        results.update(
            {
                f"{emotion_cls}_f1"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_f1": f1
                for emotion_cls, f1 in zip(
                    data_loader.dataset.emotions, f1_scores
                )
            }
        )

        results.update(
            {
                f"{emotion_cls}_roc"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_roc": roc
                for emotion_cls, roc in zip(
                    data_loader.dataset.emotions, roc_auc_scores
                )
            }
        )

        results.update(
            {
                f"{emotion_cls}_f1_max"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_f1_max": f1
                for emotion_cls, f1 in zip(
                    data_loader.dataset.emotions, max_f1_scores
                )
            }
        )    

        results.update(
            {
                f"{emotion_cls}_threshold_max"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_threshold_max": threshold
                for emotion_cls, threshold in zip(
                    data_loader.dataset.emotions, max_f1_thresholds
                )
            }
        )     

        reddit_true = np.array(eval_true)[self.dev_dataset.platform_idx]
        reddit_preds = np.array(eval_preds)[self.dev_dataset.platform_idx]
        reddit_probs = np.array(eval_probs)[self.dev_dataset.platform_idx]

        reddit_f1_scores = f1_score(
            reddit_true, reddit_preds, average=None, zero_division=1
        )

        reddit_roc_auc = roc_auc_score(reddit_true, reddit_probs, average=None)

        reddit_max_f1s = []
        reddit_f1_thresholds = []
        
        for true, probs in zip(np.array(reddit_true).T, np.array(reddit_probs).T):
            max_f1 = 0
            max_threshold = 0
            for threshold in np.arange(0, 1, 0.05):
                preds = (probs > threshold).astype('int')
                f1 = f1_score(true, preds)
                if f1 > max_f1:
                    max_f1 = f1
                    max_threshold = threshold
            
            reddit_max_f1s.append(max_f1)
            reddit_f1_thresholds.append(max_threshold)

        results.update(
            {
                f"{emotion_cls}_f1_reddit"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_f1_reddit": f1
                for emotion_cls, f1 in zip(
                    data_loader.dataset.emotions, reddit_f1_scores
                )
            }
        )

        results.update(
            {
                f"{emotion_cls}_roc_reddit"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_roc_reddit": roc
                for emotion_cls, roc in zip(
                    data_loader.dataset.emotions, reddit_roc_auc
                )
            }
        )

        results.update(
            {
                f"{emotion_cls}_f1_max_reddit"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_f1_max_reddit": f1
                for emotion_cls, f1 in zip(
                    data_loader.dataset.emotions, reddit_max_f1s
                )
            }
        )    

        results.update(
            {
                f"{emotion_cls}_threshold_max_reddit"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_threshold_max_reddit": threshold
                for emotion_cls, threshold in zip(
                    data_loader.dataset.emotions, reddit_f1_thresholds
                )
            }
        )     


        return results

    def get_eval_preds_from_batch(
        self, logits: torch.Tensor
    ) -> List[List[int]]:
        """Transforms logits to list of predictions."""
        return (logits.sigmoid() > 0.5).int().tolist()

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> List[List[int]]:
        """Transforms labels to list."""
        return labels.tolist()


class SinglelabelEmotionTrainer(BaseTrainer):
    """Base trainer class for Single-label Emotion Recognition.

    Assumption here is that the outputs of the dataset
    conform to `transformers` conventions, i.e. the first output
    is expected to be a dictionary.

    Attributes:
        Check `BaseTrainer`.
    """

    def train_end(self):
        self.exp_handler.log()
        self._save_best_model()
        self.exp_handler.aggregate_results()
        self.exp_handler.plot(
            groups=[
                [
                    f"{emotion_cls}_f1"
                    if isinstance(emotion_cls, str)
                    else "-".join(emotion_cls) + "_f1"
                    for emotion_cls in self.dev_dataset.emotions
                ]
            ]
            if self.do_eval
            else None
        )

    def batch_len(self, batch):
        """Returns length of tensors inside inputs dict."""
        return len(next(iter(batch[0].values())))

    def input_batch_kwargs(self, batch):
        """Returns input dict."""
        return batch[0]

    def evaluation_metrics(
        self,
        eval_true: List[List[int]],
        eval_preds: List[List[int]],
        eval_probs: List[List[float]],
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """Computes evaluation F1s and JS.

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.

        Returns:
            A dict of metrics.
        """

        macro_f1 = f1_score(
            eval_true, eval_preds, average="macro", zero_division=1
        )
        eval_accuracy = f1_score(
            eval_true, eval_preds, average="micro", zero_division=1
        )

        f1_scores = f1_score(
            eval_true, eval_preds, average=None, zero_division=1
        )

        results = dict(eval_accuracy=eval_accuracy, macro_f1=macro_f1)
        results.update(
            {
                f"{emotion_cls}_f1"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_f1": f1
                for emotion_cls, f1 in zip(
                    data_loader.dataset.emotions, f1_scores
                )
            }
        )

        return results

    def get_eval_preds_from_batch(self, logits: torch.Tensor) -> List[int]:
        """Transforms logits to list of predictions."""
        return logits.argmax(-1).tolist()

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> List[int]:
        """Transforms labels to list."""
        return labels.tolist()


class SemEval2018Task1EcTrainer(MultilabelEmotionTrainer):
    """Generic trainer for SemEval Task 1 E-c.

    Attributes:
        Check `MultilabelEmotionTrainer`.
        bce_loss_weighter: `MultilabelConditionalWeights` module to get
            weight for each bce loss.
        label_correlations: `Correlations` module to get weights for each
            pair of emotions in a local loss.
    """

    argparse_args = deepcopy(MultilabelEmotionTrainer.argparse_args)
    _function_choice_str = (
        "choose between 'identity', 'sqrt', 'square' and 'log', "
        "append a '_p1' to add 1"
    )

    argparse_args.update(
        dict(
            local_correlation_coef=dict(
                type=float, help="local correlation loss coefficient"
            ),
            multilabel_conditional_order=dict(
                type=int,
                help="order of relations to model between labels for conditional "
                "weighting scheme",
            ),
            multilabel_conditional_func=dict(
                type=str,
                help="what function of conditional probability to use, "
                + _function_choice_str,
            ),
            local_correlation_weighting_func=dict(
                type=str,
                help="what function of correlation of pair to use in local correlation loss, "
                + _function_choice_str,
            ),
            local_correlation_loss=dict(
                type=str,
                help="what local correlation loss function to use",
                default="inter_exp_diff",
            ),
            local_correlation_priors=dict(
                action="store_true",
                help="whether to use prior correlations rather than "
                "data-driven ones",
            ),
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bce_loss_weighter = MultilabelConditionalWeights(
            self.exp_handler.multilabel_conditional_order,
            func=self.exp_handler.multilabel_conditional_func,
        )

        dataset_labels = self.dataset.labels
        if isinstance(dataset_labels, list):
            dataset_labels = torch.cat(dataset_labels)

        self.bce_loss_weighter.fit(dataset_labels)
        self.label_correlations = Correlations(
            dataset_labels
            if not self.exp_handler.local_correlation_priors
            else None,
            self.dataset.all_emotions,
            func=self.exp_handler.local_correlation_weighting_func,
            active=self.exp_handler.local_correlation_weighting,
        )

    def train_init(self):
        """Used when training starts. Loads local checkpoint."""
        super().train_init()
        if self.exp_handler.model_load_filename is not None:
            state_dict = torch.load(self.exp_handler.model_load_filename)
            self.model.load_state_dict(
                state_dict,
                discard_classifier=self.exp_handler.discard_classifier,
            )

    _inter_distances = dict(
        sq_diff=lambda p, a, w: -(p - a.unsqueeze(-1)).square().mul(w).mean(),
        cossim=lambda p, a, w: cosine_similarity(
            p.unsqueeze(-1), a.unsqueeze(0).transpose(2, 1), dim=1
        )
        .mul(w)
        .mean(),
        exp_diff=lambda p, a, w: (a - p.unsqueeze(-1)).exp().mul(w).mean(),
    )

    _intra_distances = dict(
        sq_diff=lambda x, w: (x - x.unsqueeze(-1))
        .triu(diagonal=1)
        .square()
        .mul(w)
        .mean(),
        cossim=lambda x, w: -cosine_similarity(
            x.unsqueeze(-1), x.unsqueeze(0).transpose(2, 1), dim=1
        )
        .triu(diagonal=1)
        .mul(w)
        .mean(),
        exp_diff=lambda x, w: (x + x.unsqueeze(-1))
        .triu(diagonal=1)
        .exp()
        .mul(w)
        .mean(),
    )

    def _inter_correlation(
        self,
        vals: torch.Tensor,
        trues: torch.Tensor,
        categories: List[str],
        distance_func: str,
    ) -> torch.Tensor:
        """Calculates local correlation loss loss for one input example.

        Args:
            vals: representations of the network to calculate loss on.
            trues: ground-truth labels.
            categories: names of labels.
            distance_func: what distance between the representations to use.
                Available are `"cossim"` for multidimensional representations
                and `"exp_diff"` for scalars.

        Returns:
            Loss.
        """

        distance_func = self._inter_distances[distance_func]

        absent_inds = trues < 0.5
        present_inds = trues >= 0.5

        # check if at least one element in both groups
        if not any(absent_inds) or not any(present_inds):
            return torch.tensor(0.0, device=vals.device)

        absent = vals[absent_inds].sigmoid()
        present = vals[present_inds].sigmoid()

        weight = self.label_correlations.get(
            (
                categories[absent_inds.cpu()].tolist(),
                categories[present_inds.cpu()].tolist(),
            ),
            decreasing=True,
        )

        if weight is None:
            weight = torch.ones(present.shape[0], absent.shape[0])

        return distance_func(
            present, absent, weight.to(self.exp_handler.device)
        )

    def _intra_correlation(
        self,
        vals: torch.Tensor,
        trues: torch.Tensor,
        categories: List[str],
        distance_func: str,
    ) -> torch.Tensor:
        """Calculates local correlation loss loss but for same group predictions for one example."""

        distance_func = self._intra_distances[distance_func]

        example_loss = torch.tensor(0.0, device=vals.device)

        absent_inds = trues < 0.5
        present_inds = trues >= 0.5

        if any(absent_inds):
            absent = vals[absent_inds]
            absent_emotions = categories[absent_inds.cpu()].tolist()

            weight = self.label_correlations.get(
                (absent_emotions, absent_emotions), decreasing=False
            )

            if weight is None:
                weight = torch.ones(absent.shape[0], absent.shape[0])

            example_loss = example_loss + distance_func(
                absent, weight.to(self.exp_handler.device)
            )

        if any(present_inds):
            present = vals[present_inds]
            present_emotions = categories[present_inds.cpu()].tolist()

            weight = self.label_correlations.get(
                (present_emotions, present_emotions), decreasing=False
            )

            if weight is None:
                weight = torch.ones(
                    present.shape[0],
                    present.shape[0],
                    device=self.exp_handler.device,
                )

            ## - for exp_diff loss, cossim handles it just fine
            example_loss = example_loss + distance_func(
                -present, weight.to(self.exp_handler.device)
            )

        if any(present_inds) and any(absent_inds):
            example_loss = example_loss / 2

        return example_loss

    def _complete_correlation(
        self,
        vals: torch.Tensor,
        trues: torch.Tensor,
        categories: List[str],
        distance_func: str,
    ) -> torch.Tensor:
        """Calculates average of local correlation loss losses for inter and intra groups
        for one example."""
        inter = self._inter_correlation(vals, trues, categories, distance_func)
        intra = self._intra_correlation(vals, trues, categories, distance_func)
        if inter == 0:
            return intra

        return (inter + intra) / 2

    def calculate_cls_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, train: bool
    ) -> torch.Tensor:
        if train:
            weights = self.bce_loss_weighter.get_batch_weights(
                self.get_eval_true_from_batch(labels),
                self.get_eval_preds_from_batch(logits),
            )
            criterion = nn.BCEWithLogitsLoss(
                weight=weights.to(self.exp_handler.device)
            )
        else:
            criterion = nn.BCEWithLogitsLoss()

        bce_loss = criterion(logits, labels)
        if self.exp_handler.local_correlation_coef:  # coef not None and > 0
            bce_loss = (1 - self.exp_handler.local_correlation_coef) * bce_loss

        return bce_loss

    def calculate_regularization_loss(
        self,
        intermediate_representations: Optional[torch.Tensor],
        logits: torch.Tensor,
        batch: Iterable[Any],
        train: bool,
    ) -> torch.Tensor:

        if train:
            emotions = np.array(self.dataset.emotions, dtype=object)
        else:
            emotions = np.array(self.dataset.all_emotions, dtype=object)

        (
            loss_type,
            *distance_func,
        ) = self.exp_handler.local_correlation_loss.split("_")
        distance_func = "_".join(distance_func)

        loss = torch.tensor(0.0, device=self.exp_handler.device)

        if self.exp_handler.local_correlation_coef:  # coef not None and > 0
            local_corr_loss_func = self.__getattribute__(
                f"_{loss_type}_correlation"
            )

            # bcs intermediate_representations is optional
            if intermediate_representations is not None:
                reprs = (
                    logits
                    if distance_func == "exp_diff"
                    else intermediate_representations
                )
            else:
                reprs = logits
                distance_func = "exp_diff"

            local_corr_loss = torch.stack(
                [
                    local_corr_loss_func(vals, true, emotions, distance_func)
                    for vals, true in zip(reprs, self.batch_labels(batch))
                ]
            ).mean()
            loss = (
                loss + self.exp_handler.local_correlation_coef * local_corr_loss
            )

        return loss

    def batch_ids(self, batch: Iterable[Any]):
        """Returns some identifier for the examples of the batch."""
        if len(batch) > 2:
            return batch[2]

    def evaluation_metrics(
        self,
        eval_true: List[int],
        eval_preds: List[int],
        eval_probs: List[float],
        data_loader: DataLoader,
        eval_ids: Optional[List[str]],
    ) -> Dict[str, float]:

        results = super().evaluation_metrics(eval_true, eval_preds, eval_probs, data_loader)
        # original_metrics = list(results.keys())

        if eval_ids:

            eval_groups = {}
            for t, p, o, i in zip(eval_true, eval_preds, eval_probs, eval_ids):
                eval_groups.setdefault(i, {}).setdefault(
                    "eval_true", []
                ).append(t)
                eval_groups.setdefault(i, {}).setdefault(
                    "eval_preds", []
                ).append(p)
                eval_groups.setdefault(i, {}).setdefault(
                    "eval_probs", []
                ).append(o)

            for i in eval_groups:
                results.update(
                    {
                        f"{i}_{k}": v
                        for k, v in super()
                        .evaluation_metrics(
                            eval_groups[i]["eval_true"],
                            eval_groups[i]["eval_preds"],
                            eval_groups[i]["eval_probs"],
                            data_loader,
                        )
                        .items()
                    }
                )

            # for k in original_metrics:
            # get metric per lang
            metrics_lang = {}
            # get support per lang
            lang_support = Counter(eval_ids)
            for lang_key in results:
                lang, *old_key = lang_key.split("_")
                if lang in eval_groups:
                    old_key = "_".join(old_key)
                    metrics_lang.setdefault(old_key, []).append(
                        (results[lang_key], lang_support[lang])
                    )

            # compute macro, micro per lang
            for old_key in metrics_lang:
                results.update(
                    {
                        f"weighted_{old_key}": sum(
                            [s * m for m, s in metrics_lang[old_key]]
                        )
                        / sum([s for _, s in metrics_lang[old_key]])
                    }
                )
                results.update(
                    {
                        f"macro_{old_key}": sum(
                            [m for m, _ in metrics_lang[old_key]]
                        )
                        / len(metrics_lang[old_key])
                    }
                )

        return results


class GoEmotionsTrainer(SemEval2018Task1EcTrainer):
    """Generic trainer for GoEmotions. Check `SemEval2018Task1EcTrainer`."""

    early_stopping_metric = "micro_f1"

    def evaluation_metrics(
        self,
        eval_true: List[List[int]],
        eval_preds: List[List[int]],
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        (
            micro_precision,
            micro_recall,
            micro_f1,
            _,
        ) = precision_recall_fscore_support(
            eval_true, eval_preds, average="micro", zero_division=0
        )
        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            eval_true, eval_preds, average="macro", zero_division=0
        )
        f1_scores = f1_score(
            eval_true, eval_preds, average=None, zero_division=0
        )
        results = dict(
            micro_f1=micro_f1,
            micro_recall=micro_recall,
            micro_precision=micro_precision,
            macro_f1=macro_f1,
            macro_recall=macro_recall,
            macro_precision=macro_precision,
        )
        results.update(
            {
                f"{emotion_cls}_f1"
                if isinstance(emotion_cls, str)
                else "-".join(emotion_cls) + "_f1": f1
                for emotion_cls, f1 in zip(
                    data_loader.dataset.emotions, f1_scores
                )
            }
        )

        return results


class FrenchElectionTrainer(SemEval2018Task1EcTrainer):
    """Generic trainer for French Election data. Check `SemEval2018Task1EcTrainer`."""

class MHSTrainer(SemEval2018Task1EcTrainer):
    """Generic trainer for Hate speech data. Check `SemEval2018Task1EcTrainer`."""

class AnnotatorTrainer(SemEval2018Task1EcTrainer):
    """Generic trainer for Hate speech data w/annotators. Check `SemEval2018Task1EcTrainer`."""
    def input_batch_kwargs(self, batch):
        """Returns input dict."""
        return batch[:-1]

    def batch_labels(self, batch: Iterable[Any]):
        return batch[-1]

    def evaluate(
        self,
        data_loader: DataLoader,
        tqdm_message: Optional[str] = "Evaluation",
    ):
        """Evaluates model on `data_loader`.

        Args:
            data_loader: dataset to evaluate on.
            tqdm_message: what to print if tqdm is used.
        """

        self.model.eval()
        self.eval_init(data_loader)

        batch_itr = (
            tqdm(data_loader, desc=tqdm_message, dynamic_ncols=True)
            if not self.exp_handler.disable_tqdm
            and not isinstance(
                data_loader.dataset, torch.utils.data.IterableDataset
            )
            else data_loader
        )

        eval_preds = []
        eval_true = []
        eval_probs = []
        eval_ids = []
        eval_loss = 0.0
        eval_reg_loss = 0.0

        for batch in batch_itr:
            batch = self.batch_to_device(batch)

            with torch.no_grad():
                inputs_batch = self.input_batch_kwargs(batch)
                return_vals = self.model(inputs_batch[1], **inputs_batch[0])

            logits = self.get_logits_from_model(return_vals, batch, data_loader)
            inter_reprs = self.get_intermediate_repr_from_model(
                return_vals, batch
            )

            _, cls_loss, reg_loss = self.calculate_loss(
                logits,
                batch,
                train=False,
                intermediate_representations=inter_reprs,
            )

            eval_loss += cls_loss.item() * self.batch_len(batch)
            eval_reg_loss += reg_loss.item() * self.batch_len(batch)

            batch_preds = logits.sigmoid() > 0.5
            batch_probs = logits.sigmoid()
            batch_true = self.batch_labels(batch)

            binary_pred = (torch.sum(batch_preds, dim=0) > batch_preds.size(0) / 2).int().tolist()
            majority_true = (torch.sum(batch_true, dim=0) > batch_true.size(0) / 2).int().tolist()
            mean_probs = torch.mean(batch_probs, dim=0).tolist()
            
            eval_preds.extend([binary_pred])
            eval_true.extend(
                [majority_true]
            )
            eval_probs.extend(
                [mean_probs]
                )

            #ids = self.batch_ids(batch)
            #if ids:
            #    eval_ids.extend(ids)

        ### compute eval metrics
        #self.logger.info(f"Eval loss:  {eval_loss}")
        eval_loss /= len(data_loader.dataset)
        eval_reg_loss /= len(data_loader.dataset)

        #self.logger.info(f"Eval true:  {eval_true}")
        #self.logger.info(f"Eval preds: {eval_preds}")
        #self.logger.info(f"Eval probs:  {eval_probs}")

        results = dict(
            eval_loss=eval_loss, eval_regularization_loss=eval_reg_loss
        )
        results.update(
            self.evaluation_metrics(
                eval_true,
                eval_preds,
                eval_probs,
                data_loader=data_loader,
                eval_ids=eval_ids,
            )
        )

        self.model.train()
        self.eval_end(data_loader)

        return results

    def train(self):
        """Trains and, if a dev set or test set was provided, evaluates
        the model. (Optimizer is without bias correction)."""

        self.model = self.model.to(self.exp_handler.device)
        self.model.train()
        self.train_init()

        kwargs = (
            dict(shuffle=True)
            if not isinstance(self.dataset, torch.utils.data.IterableDataset)
            else dict()
        )

        data_loader = DataLoader(
            self.dataset,
            batch_size=self.exp_handler.train_batch_size,
            collate_fn=getattr(self.dataset, "collate_fn", None),
            num_workers=self.exp_handler.dataloader_num_workers,
            **kwargs,
        )

        if self.do_eval:
            dev_data_loader = DataLoader(
                self.dev_dataset,
                collate_fn=getattr(self.dev_dataset, "collate_fn", None),
                num_workers=self.exp_handler.dataloader_num_workers,
                batch_sampler=self.dev_dataset.eval_batches
            )
        if self.do_test:
            test_data_loader = DataLoader(
                self.test_dataset,
                collate_fn=getattr(self.test_dataset, "collate_fn", None),
                num_workers=self.exp_handler.dataloader_num_workers,
                batch_sampler = self.test_dataset.eval_batches
            )

        self.optimizer, self.scheduler = self.init_optimizer_scheduler(
            len(data_loader)
        )

        num_epochs = int(self.exp_handler.num_train_epochs or 1)
        early_stop = False

        for epoch in range(num_epochs):

            if early_stop:
                # the early stopping check only breaks from inner loop
                break

            batch_itr = (
                tqdm(
                    data_loader,
                    desc=f"Training Epoch {epoch+1}",
                    dynamic_ncols=True,
                )
                if not self.exp_handler.disable_tqdm
                and not isinstance(
                    data_loader.dataset, torch.utils.data.IterableDataset
                )
                else data_loader
            )

            for step, batch in enumerate(batch_itr):

                step += epoch * len(data_loader)

                early_stop = (
                    self.exp_handler.max_steps > -1
                    and step >= self.exp_handler.max_steps
                )
                if early_stop:
                    self.logger.info("Forcibly stopping training")
                    break

                if step % self.exp_handler.eval_steps == 0:
                    # FIRST step of current collection of evaluation steps
                    # will always init when epoch == 0, step == 0
                    train_loss = 0.0
                    cum_regularization_loss = 0.0
                    n_samples = 0

                batch = self.batch_to_device(batch)

                self.pre_step_actions()
                
                #return_vals = self.model(**self.input_batch_kwargs(batch))
                inputs_batch = self.input_batch_kwargs(batch)
                return_vals = self.model(inputs_batch[1], **inputs_batch[0])
                logits = self.get_logits_from_model(
                    return_vals, batch, data_loader, epoch
                )
                inter_repr = self.get_intermediate_repr_from_model(
                    return_vals, batch
                )

                loss, cls_loss, reg_loss = self.calculate_loss(
                    logits,
                    batch,
                    train=True,
                    intermediate_representations=inter_repr,
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.post_step_actions()

                train_loss += cls_loss.item() * self.batch_len(batch)
                #self.logger.info(f"Train loss: {train_loss}")
                
                '''
                if math.isnan(train_loss):
                    self.logger.info(f"The culprit? {logits}")
                    self.logger.info(f"Sneaky Byatch: {batch}")
                    self.logger.info(f"Batch length? {self.batch_len(batch)}")
                    self.logger.info(f"cls_loss: {cls_loss.item()}")

                '''
                cum_regularization_loss += reg_loss.item() * self.batch_len(
                    batch
                )
                n_samples += self.batch_len(batch)

                if (step + 1) % self.exp_handler.eval_steps == 0:
                    # LAST step of current collection of evaluation steps

                    train_loss /= n_samples
                    cum_regularization_loss /= n_samples

                    results = dict(
                        train_loss=train_loss,
                        regularization_loss=cum_regularization_loss,
                    )

                    if self.do_eval:
                        results.update(
                            self.evaluate(
                                dev_data_loader,
                                f"Evaluating after {step+1} steps (epoch {epoch+1})",
                            )
                        )

                    self.exp_handler.set_dict_metrics(results)

                    self.logger.info(
                        f"Step {step+1} (epoch {epoch+1}) metrics on "
                        + self.eval_dataset_names
                        + ": "
                        + result_str(results)
                    )

                    early_stop = self.early_stopping.step(
                        results.get(self.early_stopping_metric, None),
                        **{**results, "epoch": epoch + 1, "step": step + 1},
                    )
                    if early_stop:
                        self.logger.info(
                            "Early stopping at step "
                            f"{step + 1} (epoch {epoch+1})"
                        )
                        break

        early_stopping_metrics = self.early_stopping.get_metrics()
        if early_stopping_metrics is not None:
            self.logger.info(
                f"Best metrics based on {self.early_stopping_metric} "
                f"on {self.eval_dataset_names}: "
                + result_str(early_stopping_metrics)
            )
            self.exp_handler.set_best(
                "early_stopping",
                metric=self.early_stopping_metric,
                higher_better=True,
            )

        if self.do_test:
            results = self.evaluate(test_data_loader, "Testing")
            self.logger.info(
                f"Testing metrics for {self.test_dataset_names}: "
                + result_str(results)
            )
            self.exp_handler.set_dict_metrics(results, test=True)

        self.train_end()
