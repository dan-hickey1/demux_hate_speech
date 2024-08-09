import argparse
import json
import os
from time import time
from types import SimpleNamespace
from typing import Callable, Iterable, List, Dict, Union, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from emorec.utils import flatten_list, twitter_preprocessor, reddit_preprocessor
from emorec.models.demux.annotator_demux import DemuxAnno
from emorec.models.demux.utils import demojizer_selector

Text = Dict[str, str]
Annotation = Dict[str, Union[Dict[str, float], str]]


class TextDataset(Dataset):
    """Dataset class for Demux (aka handles its prompts).

    Attributes:
        platform: platform model is going to used.
        preprocessor: text preprocessor.
        tokenizer: text tokenizer.
        emotions: list of emotions or emotion clusters.
        language: text/model language (can be `"multilingual"`).
        ids: text IDs.
        inputs: `transformer`-style inputs.
    """

    lang_disjunctions = {
        "english": "or",
        "french": "ou",
        "spanish": "o",
        "arabic": "أو",
        "multilingual": "or",
    }

    def __init__(
        self,
        texts: List[Text],
        emotions: Dict[str, Union[str, List[str]]],
        tokenizer: PreTrainedTokenizerBase,
        preprocessor: Callable,
        language: str,
        max_length: Optional[str] = None,
        platform: str = "twitter",
    ):
        """Init.

        Args:
            texts: input text.
            emotions: list of emotions or emotion clusters.
            tokenizer: text tokenizer.
            platform: platform model is going to used.
            preprocessor: text preprocessor.
            language: text/model language (can be `"multilingual"`).
            max_length: tokenization length, estimated automatically if not
                provided (default).
            platform: platform model is going to used.
        """

        self.platform = platform.lower()
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.emotion_struct = emotions
        self.emotions = list(emotions.values())
        self.language = language.lower()

        emotions_for_prompt = flatten_list(self.emotions)
        prompt = (
            " ".join(emotions_for_prompt[:-1])
            + " "
            + self.lang_disjunctions[self.language]
            + " "
            + emotions_for_prompt[-1]
        )

        self.texts = texts

        self.inputs = (tokenizer.batch_encode_plus(
            [
                (prompt, self.preprocessor(text["text"] or ""))
                for text in self.texts
            ],
            max_length=max_length or 256,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
        ), 
            torch.stack([text['annotator_metrics'] for text in self.texts]))

        self.class_inds = self._get_class_inds()

    def __getitem__(self, index: int):
        o = self.texts[index]["id"], {
            k: v[index] for k, v in self.inputs[0].items()
        }, self.inputs[1][index]
        return o

    def __len__(self):
        return len(self.texts)

    def collate_fn(
        self, batch: Iterable[Tuple[str, Dict[str, torch.Tensor]]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns multiple examples in the `transformers` format
        and their labels."""
        ids, text, annotators = [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]
        return ids, {k: torch.stack([d[k] for d in text]) for k in text[0]}, torch.stack(annotators)

    def _get_class_inds(self) -> List[torch.Tensor]:
        """Get indices of tokens for each emotion.

        Returns:
            A list of tensors, each containing all the indices
            for a specific emotion (in the order of `self.emotions`).
        """

        class_ids = [
            self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(emotion_cls)
            )
            if isinstance(emotion_cls, str)
            # else cluster of emotions
            else flatten_list(
                [
                    self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(emotion)
                    )
                    for emotion in emotion_cls
                ]
            )
            for emotion_cls in self.emotions
        ]

        example_input_ids = self.inputs[0]["input_ids"][0].tolist()

        class_inds = []
        for ids in class_ids:
            inds = []
            for _id in ids:
                id_idx = example_input_ids.index(_id)
                # in case it exists multiple times, turn identified instance
                # to None (i.e. pop w/o changing indices because indices are
                # what we are collecting)
                example_input_ids[id_idx] = None
                inds.append(id_idx)

            class_inds.append(torch.tensor(inds, dtype=torch.long))

        return class_inds


def format_output(
    texts: List[Text],
    scores: Dict[str, Dict[str, float]],
    id_column: str,
    text_column: str,
) -> List[Annotation]:
    """Formats the output scores.

    Args:
        texts: list of texts with their IDs.
        scores: score per emotion in each text.
        id_column: name of the column containing the IDs.
        text_column: name of the column containing the text.

    Returns:
        A dict whose keys are message IDs and values dicts with
        example info. Annotations are dict of emotions and their
        scores.
    """

    annotations = []

    for text in texts:
        _id = text["id"]
        annotations.append(
            {
                id_column: _id,
                text_column: text["text"],
                "annotator_id":text['annotator_id'],
                "emotions": {emo: score for emo, score in scores[_id].items()},
            }
        )

    return annotations


PLATFORM_PREPROCESSORS = {
    "twitter": twitter_preprocessor,
    "reddit": reddit_preprocessor,
    "facebook": twitter_preprocessor,
}


def pipeline_setup(
    model_name: str,
    pretrained_fn: str,
    device: str,
    annotators: list,
    platform: str = "Twitter",
) -> SimpleNamespace:
    """Returms the `model`, the `tokenizer`, and the text `preprocessor`
    in a namespace.

    Args:
        pretrained_fn: local path to pretrained model.
        device: which device to use.
        language: for which language is the model.
        platform: what social media platform is the model going to be used.
    """

    platform = platform.lower()
    demojizer = demojizer_selector(model_name)
    platform_preprocessor = PLATFORM_PREPROCESSORS[platform]()
    preprocessor = lambda x: platform_preprocessor(demojizer(x))

    model = DemuxAnno.from_pretrained(model_name, class_inds=None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    annotator_tokens = []
    for anno_id in annotators:
        token = '[' + str(anno_id) + ']'
        annotator_tokens.append(token)
    
    tokenizer.add_tokens(annotator_tokens)

    model.bert.resize_token_embeddings(len(tokenizer))


    print(f"Loading model from {pretrained_fn}")
    state_dict = torch.load(pretrained_fn, map_location="cpu")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict.pop(
            "bert.embeddings.position_ids", None
        )  # difference in transformers/python versions (??)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return SimpleNamespace(
        model=model, tokenizer=tokenizer, preprocessor=preprocessor
    )


def data_setup(
    texts: Union[List[Text], Text],
    emotions: Dict[str, str],
    pipeline: SimpleNamespace,
    language: str = "Multilingual",
    platform: str = "Twitter",
    batch_size: int = 256,
    max_length: Optional[int] = None,
) -> DataLoader:
    """Returns a DataLoader containing inputs for Demux and their IDs.


    Args:
        texts: list of `Text`, or a Text.
        emotions: dictionary with a key per emotion and a value which
            is also a dict that contains the text for the emotion to
            be used and the ontology field.
        pipeline: pipeline Namespace.
        language: for which language is the model.
        platform: what social media platform is the model going to be used.
        batch_size: inference batch size.
        max_length: tokenization length, estimated automatically if not
            provided (default).
    """
    if not isinstance(texts, list):
        texts = [texts]

    dataset = TextDataset(
        texts,
        emotions,
        pipeline.tokenizer,
        pipeline.preprocessor,
        language,
        max_length,
        platform,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=dataset.collate_fn
    )
    return dataloader


def annotate(
    pipeline: SimpleNamespace,
    dataloader: DataLoader,
    device: str,
    id_column: str,
    text_column: str,
) -> Dict[str, List[Annotation]]:
    """Annotates text in `dataloader` through the `pipeline`.

    Args:
        pipeline: pipeline Namespace.
        dataloader: loader containing the IDs and the text.
        device: which device to use.

    Returns:
        A list with one dict per tweet of the form:
        {
            "id": "swqeivfbo23ierg74befiod",
            "text": "Fear, Pessimism",
            "confidence": 0.34553456,
            "providerName": "ta1-usc-isi",
        }
    """

    pipeline.model.set_class_inds(dataloader.dataset.class_inds)

    confidences = {}
    confidence_fn = nn.Sigmoid()

    for batch_ids, batch_inputs, batch_annotators in tqdm(
        dataloader, desc="Annotating in batches"
    ):
        batch_inputs = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }

        batch_annotators = batch_annotators.to(device)

        with torch.no_grad():
            out, _ = pipeline.model(batch_annotators, **batch_inputs)
            scores = confidence_fn(out)

        for batch_id, batch_scores in zip(batch_ids, scores):
            confidences[batch_id] = {
                emo: confidence
                for emo, confidence in zip(
                    dataloader.dataset.emotion_struct,
                    batch_scores.to("cpu").tolist(),
                )
            }

    return format_output(
        dataloader.dataset.texts, confidences, id_column, text_column
    )


def to_dict(obj, classkey=None):
    if isinstance(obj, dict):
        data = {}
        for k, v in obj.items():
            data[k] = to_dict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return to_dict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [to_dict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict(
            [
                (key, to_dict(value, classkey))
                for key, value in obj.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-folder", required=True, type=str, help="model folder"
    )
    parser.add_argument(
        "--emotion-config",
        required=True,
        type=str,
        help="json filename for emotion names and corresponding DEMUX classes",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="twitter",
        help="source domain of data",
        choices=["twitter", "reddit", "facebook"],
    )
    parser.add_argument("--device", type=str, help="gpu/cpu to use")
    parser.add_argument(
        "--input-filename",
        type=str,
        nargs="+",
        help="which file(s) to load from",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["json", "jsonl", "csv"],
        help="format of input file(s), if different from extension",
    )
    parser.add_argument(
        "--out-filename", required=True, type=str, help="where to save file"
    )
    parser.add_argument(
        "--batch-size", type=int, help="number of texts in batch", default=256
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="max length of text to tokenize",
    )
    parser.add_argument(
        "--id-column", type=str, default="id", help="column name for id in csv"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="column name for text in csv/json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_format = (
        args.input_format or os.path.splitext(args.input_filename[0])[1][1:]
    )

    device = (
        args.device
        if torch.cuda.is_available() and args.device is not None
        else "cpu"
    )

    annotator_cols = ['annotator_gender_men', 'annotator_gender_women',
       'annotator_gender_non_binary', 'annotator_gender_prefer_not_to_say',
       'annotator_gender_self_describe', 'annotator_transgender',
       'annotator_cisgender', 'annotator_transgender_prefer_not_to_say',
       'annotator_education_some_high_school',
       'annotator_education_high_school_grad',
       'annotator_education_some_college',
       'annotator_education_college_grad_aa',
       'annotator_education_college_grad_ba',
       'annotator_education_professional_degree',
       'annotator_education_masters', 'annotator_education_phd',
       'annotator_income_<10k', 'annotator_income_10k-50k',
       'annotator_income_50k-100k', 'annotator_income_100k-200k',
       'annotator_income_>200k', 'annotator_ideology_extremeley_conservative',
       'annotator_ideology_conservative',
       'annotator_ideology_slightly_conservative',
       'annotator_ideology_neutral', 'annotator_ideology_slightly_liberal',
       'annotator_ideology_liberal', 'annotator_ideology_extremeley_liberal',
       'annotator_ideology_no_opinion', 'annotator_race_asian',
       'annotator_race_black', 'annotator_race_latinx',
       'annotator_race_middle_eastern', 'annotator_race_native_american',
       'annotator_race_pacific_islander', 'annotator_race_white',
       'annotator_race_other', 'annotator_religion_atheist',
       'annotator_religion_buddhist', 'annotator_religion_christian',
       'annotator_religion_hindu', 'annotator_religion_jewish',
       'annotator_religion_mormon', 'annotator_religion_muslim',
       'annotator_religion_nothing', 'annotator_religion_other',
       'annotator_sexuality_bisexual', 'annotator_sexuality_gay',
       'annotator_sexuality_straight', 'annotator_sexuality_other', 'annotator_age']

    mhs = pd.read_csv('./mhs_unaggregated_annotators.csv')

    mhs = mhs.drop_duplicates(subset='annotator_id')[['annotator_id'] + annotator_cols].reset_index(drop=True).set_index('annotator_id')
    annotators = mhs.index.to_list()

    with open(os.path.join(args.pretrained_folder, "config.json")) as fp:
        config = json.load(fp)

    pipeline = pipeline_setup(
        config["model_name"],
        os.path.join(args.pretrained_folder, "model.pt"),
        device,
        annotators,
        args.domain,
    )

    with open(args.emotion_config) as fp:
        emotion_config = json.load(fp)
    for emotion in emotion_config:
        if isinstance(emotion_config[emotion], str):
            emotion_config[emotion] = [emotion_config[emotion]]

    annotations = []
    for fn in args.input_filename:
        print(f"Reading {fn}...", end=" ")
        with open(fn) as fp:
            if input_format == "jsonl":
                messages = pd.read_json(fp, lines=True)
                messages = [
                    dict(
                        id=message[args.id_column],
                        text=message[args.text_column],
                    )
                    for _, message in messages.iterrows()
                ]
            elif input_format == "json":
                messages = json.load(fp)
                messages = [dict(id=k, text=v) for k, v in messages.items()]
            elif input_format == "csv":
                text_df = pd.read_csv(fp, lineterminator='\n')

                messages = []
                for _, message in text_df.iterrows():
                    rand_annotators = np.random.choice(mhs.index, 5, replace=False)
                    for rand_annotator in rand_annotators:
                        annotator_metrics = torch.from_numpy(mhs.loc[rand_annotator].values).float()
                        messages.append(dict(
                                            id=str(message[args.id_column]) + '_' + str(rand_annotator),
                                            text=f'[{rand_annotator}] </s> {message[args.text_column]}',
                                            annotator_metrics=annotator_metrics,
                                            annotator_id=str(rand_annotator)
                        ))
                
            print("Done")

            if messages:
                t0 = time()
                data = data_setup(
                    messages,
                    emotion_config,
                    pipeline,
                    platform=args.domain,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )

                new_annotations = to_dict(
                    annotate(
                        pipeline,
                        data,
                        device,
                        args.id_column,
                        args.text_column,
                    )
                )
                annotations.extend(new_annotations)
                print(f"Annotation time: {time()-t0:.3f} sec")

            else:
                print("No input found in file")

    if annotations:
        print(f"Saving to {args.out_filename}")
        with open(args.out_filename, "w") as fp:
            #print(annotations)
            for annotation in annotations:
                fp.write(json.dumps(annotation) + "\n")


if __name__ == "__main__":
    main()
