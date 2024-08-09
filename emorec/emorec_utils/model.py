from typing import List, Dict, Union

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, AutoConfig


class BaseModel(nn.Module):
    """Demux.

    Attributes:
        bert: large LM with a BertModel-like 'interface'.
        classifier: shared FFN on top of contextual representations.
        class_inds: indices of each emotion/label in the tokenized
            input, expected to be constant in training and testing
            (because prompt goes first).
    """

    argparse_args = dict(
        model_name_or_path=dict(
            type=str,
            help="model to load into BERT parts of model",
        ),
        dropout_prob=dict(
            default=0.1,
            type=float,
            help="dropout before final linear layer",
        ),
        discard_classifier=dict(
            action="store_true",
            help="if loading a local checkpoint, "
            "whether (not) to load final classifier",
        ),
        freeze_word_embeddings=dict(
            action="store_true", help="whether to freeze LM's word embeddings"
        ),
        freeze_emotion_embeddings=dict(
            action="store_true",
            help="whether to freeze LM's emotion word embeddings",
        ),
        aggregate_logits=dict(
            action="store_true",
            help="whether to aggregate at the logit level",
        ),
    )

    def __init__(
        self,
        config: PretrainedConfig,
        aggregate_logits: bool = False,
        dropout_prob: float = 0.1,
        num_classes: int = 23
    ):
        """Init.

        Args:
            config: LM configuration from `AutoConfig`.
            class_inds: indices (`torch.long`) of each emotion/label
                in the tokenized input.
            dropout_prob: dropout before final linear layer.
        """
        super().__init__()

        self.aggregate_logits = aggregate_logits

        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob

        try:
            self.bert = AutoModel.from_config(config, add_pooling_layer=False)
        except TypeError:
            self.bert = AutoModel.from_config(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, num_classes),
        )

    def freeze_word_embeddings(self):
        self.embeddings.word_embeddings.requires_grad_(False)

    def reset_word_embeddings(self, embedding_dict):
        for i, emb in embedding_dict.items():
            self.embeddings.word_embeddings.weight.data[i] = emb.clone()

    @property
    def embeddings(self):
        return self.bert.embeddings

    @classmethod
    def from_pretrained(cls, pretrained_lm: str, *args, **kwargs) -> "Demux":
        config = AutoConfig.from_pretrained(pretrained_lm)
        model = cls(config, *args, **kwargs)
        try:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(
                    pretrained_lm, add_pooling_layer=False
                ).state_dict()
            )
        except TypeError:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(pretrained_lm).state_dict()
            )
        return model

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        discard_classifier: bool = False,
    ):
        """Loads a `state_dict`. Adds ability to discard incoming
        final classifier layers by setting `discard_classifier` to
        `True`."""

        if discard_classifier:
            state_dict = {
                ".".join(k.split(".")[1:]): v
                for k, v in state_dict.items()
                if not k.startswith("classifier")
            }
            return self.bert.load_state_dict(state_dict, strict)
        return super().load_state_dict(state_dict, strict)


    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward propagation.

        Args:
            `transformers`-style LM inputs.
            class_inds: different `class_inds` can be specified
                from the one at initialization only for this
                forward pass.

        Returns:
            Logits whose number is equal to `len(self.class_inds)`,
            contextual representation used for each is average
            of outputs specified by each list.
        """

        output = self.bert(*args, **kwargs).last_hidden_state[:,0,:]

        preds = self.classifier(output)

        return preds  # none for intermediate reprs
