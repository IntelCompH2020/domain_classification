# -*- coding: utf-8 -*-
"""
A custom classifier based on the RobertaModel class from transformers.

Created on February 2022

@author: José Antonio Espinosa
"""

import copy
import logging
from pathlib import Path
from time import time

import numpy as np
import torch
from simpletransformers.classification import ClassificationModel
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
from transformers import logging as hf_logging

# Remove message when loading transformers model
hf_logging.set_verbosity_error()


class CustomDataset(Dataset):
    def __init__(self, df):

        self.id = None
        self.text = None
        self.labels = None
        self.sample_weight = None

        for k, v in df.to_dict(orient="list").items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __getitem__(self, idx):
        item = {
            "id": self.id[idx],
            "text": self.text[idx],
        }
        if self.sample_weight is not None:
            item["sample_weight"] = self.sample_weight[idx]
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.id)


class CustomEncoderLayer(nn.Module):
    """
    Custom encoder layer of transformer for classification.
    """

    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=768,
        intermediate_size=3072,
        layer_norm_eps=1e-05,
        num_attention_heads=12,
        num_hidden_layers=1,
    ):
        super().__init__()

        # Default values
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        # Transformer encoder
        self.norm_layer = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_attention_heads,
            dim_feedforward=self.intermediate_size,
            batch_first=True,
            activation=self.hidden_act)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_hidden_layers,
            norm=self.norm_layer)

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        features: Tensor
            The sequence to the encoder
        mask: Tensor
            The mask for the src keys per batch
        """

        out = self.transformer_encoder(
            features, src_key_padding_mask=~mask.bool())

        return out


class CustomClassificationHead(nn.Module):
    """
    Copy of class RobertaClassificationHead
    (from transformers.models.roberta.modeling_roberta)
    Head for sentence-level classification tasks.
    """

    def __init__(self, classifier_dropout=0.1, hidden_dropout_prob=0.1,
                 hidden_size=768):

        super().__init__()

        # Configuration
        self.classifier_dropout = classifier_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size

        # Output labels
        self.num_labels = 2

        # Classification layers
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.classifier_dropout)
        #self.out_proj = nn.Linear(self.hidden_size, self.num_labels)
        self.out_proj_2 = nn.Linear(self.hidden_size,1)
        self.out_b = nn.Parameter(torch.empty(1,))

    def forward(self, features: torch.Tensor):
        """
        Parameters
        ----------
        features: Tensor
            The encoded text
        """

        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        #x = self.out_proj(x)
        x = self.out_proj_2(x)
        x = 1/(1+(self.out_b**2)+torch.exp(-x))
        if not x.requires_grad:
            x = x * (1 + (self.out_b) ** 2)
        x = torch.squeeze(torch.stack((x,1-x),dim=1))
        return x


class CustomModel(nn.Module):
    """
    Copy of class RobertaClassificationHead
    (from transformers.models.roberta.modeling_roberta)
    Head for sentence-level classification tasks.
    """

    def __init__(self, config, path_model):
        super().__init__()

        # Configuration
        self.config = config
        # self.classifier_dropout = config.classifier_dropout
        # self.hidden_act = config.hidden_act
        # self.hidden_dropout_prob = config.hidden_dropout_prob
        # self.hidden_size = config.hidden_size
        # self.intermediate_size = config.intermediate_size
        # self.layer_norm_eps = config.layer_norm_eps
        # self.num_attention_heads = config.num_attention_heads
        # self.num_hidden_layers = config.num_hidden_layers

        # Default values
        self.classifier_dropout = 0.1
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.layer_norm_eps = 1e-05
        self.num_attention_heads = 12
        self.num_hidden_layers = 1

        # Model location
        self.path_model = path_model

        # Transformer encoder
        self.encoderTransform = CustomEncoderLayer(
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers)

        # Classification layer
        self.classifier = CustomClassificationHead(
            classifier_dropout=self.classifier_dropout,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_size=self.hidden_size)

        # Load configuration
        self.load_embeddings()
        self.load_tokenizer()

    def create_data_loader(self, df, batch_size=8):
        """
        Creates a DataLoader from a DataFrame to train/eval model
        """

        df_set = CustomDataset(df)
        loader = DataLoader(
            dataset=df_set, batch_size=batch_size, shuffle=True, num_workers=0
        )

        return loader

    def load_tokenizer(self):
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        logging.info("-- -- Tokenizer loaded")
        # logging.info(f" -- Max length: {tokenizer.model_max_length}")
        self.tokenizer = tokenizer

    def load_embeddings(self):
        """
        Load embeddings layer.

        If there is no previous configuration, copy it from simpletransformers
        ClassificationModel and save it.
        """
        path2embeddings_state = self.path_model / "embeddings.pt"

        # Save model config if not saved before
        if not path2embeddings_state.exists():
            # Load TransformerModel
            logging.info(
                f"-- -- No available embeddings. Loading embeddings from "
                "roberta model.")
            model = ClassificationModel(
                "roberta", "roberta-base", use_cuda=False)

            embeddings = copy.deepcopy(model.model.roberta.embeddings)

            # Save model
            torch.save(embeddings.state_dict(), path2embeddings_state)
            logging.info("-- -- Embeddings model saved")

            del model

        else:
            # Load model
            embeddings = RobertaEmbeddings(self.config)
            embeddings.load_state_dict(torch.load(path2embeddings_state))
            logging.info("-- -- Embeddings model loaded from file")

        # Set grad to false to freeze layer
        for param in embeddings.parameters():
            param.requires_grad = False

        self.embeddings = embeddings

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """
        Parameters
        ----------
        features: Tensor
            The sequence to the encoder
        mask: Tensor
            The mask for the src keys per batch
        """

        out = self.encoderTransform(features, mask=mask)
        out = self.classifier(out)
        return out

    def train_model(self, df_train, device="cuda"):
        """
        Train the model

        Parameters
        ----------
        df_train: DataFrame
            Training dataframe
        epochs: int
            Number of epochs to train model
        """

        # Convert DataFrame to DataLoader
        train_data = self.create_data_loader(df_train)

        # Balance weights giving more weight to the less common label
        label_occurrences = df_train["labels"].value_counts()
        if len(label_occurrences) == 2:
            weights_train = (
                label_occurrences.max() / label_occurrences.sort_index())
            weights_train = torch.tensor(weights_train.tolist())
        else:
            weights_train = torch.tensor([1.0, 1.0])

        # Set criterion and optimizer
        criterion = nn.CrossEntropyLoss(weight=weights_train, reduction="none")
        # i could give the results in two columns from the begining and use CELoss
        #criterion = nn.BCELoss(weight=weights_train, reduction="none")  
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        # Set device
        self.to(device)
        self.embeddings.to(device)
        criterion.to(device)

        # Turn on train mode
        self.train()

        running_loss = []
        start_time = time()
        for i, data in enumerate(tqdm(train_data, desc="Train batch",
                                      leave=None)):

            # get the inputs; data is a list of [inputs, labels]
            data_id = data.get("id")
            labels = data.get("labels").to(device)
            text = data.get("text")
            sample_weight = data.get(
                "sample_weight", torch.tensor(1)).to(device)

            # Tokenize
            tokenized = self.tokenizer(
                text, padding="max_length", truncation=True)
            # print(tokenized["input_ids"])
            input_ids = torch.tensor(tokenized["input_ids"]).to(device)
            attention_mask = torch.tensor(
                tokenized["attention_mask"]).to(device)
            # Embeddings
            embs = self.embeddings(input_ids)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.forward(embs, attention_mask)
            loss = criterion(outputs, labels)
            # loss = (loss * sample_weight / sample_weight.sum()).sum()
            loss = (loss * sample_weight / len(sample_weight)).sum()
            loss.backward()
            optimizer.step()

            # Save result
            running_loss.append(loss.to("cpu").item())

        end_time = time()
        running_loss = np.sum(running_loss)
        t_time = end_time - start_time

        self.to("cpu")
        self.embeddings.to("cpu")
        criterion.to("cpu")

        return running_loss, t_time  # , batch_data

    def eval_model(self, df_eval, device="cuda"):
        """
        Evaluate trained model
        """

        # Convert DataFrame to DataLoader
        eval_data = self.create_data_loader(df_eval)

        self.to(device)
        self.embeddings.to(device)

        # turn on evaluation mode
        self.eval()
        predictions = []
        total_loss = []
        scores = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        with torch.no_grad():
            for i, data in enumerate(tqdm(eval_data, desc="Eval batch",
                                          leave=None)):
                data_id = data.get("id")
                labels = data.get("labels").to(device)
                text = data.get("text")
                sample_weight = data.get(
                    "sample_weight", torch.tensor(1)).to(device)

                # Tokenize
                tokenized = self.tokenizer(
                    text, padding="max_length", truncation=True)
                # print(tokenized["input_ids"])
                input_ids = torch.tensor(tokenized["input_ids"]).to(device)
                attention_mask = torch.tensor(
                    tokenized["attention_mask"]).to(device)
                # Embeddings
                embs = self.embeddings(input_ids)

                # Forward
                outputs = self.forward(embs, attention_mask)

                err = cross_entropy(outputs, labels)
                # err = (err * sample_weight / sample_weight.sum()).sum()
                err = (err * sample_weight / len(sample_weight)).sum()
                total_loss.append(err.to("cpu").item())

                # Save predictions
                predictions.extend(outputs.to("cpu").tolist())

                # Compute scores
                preds = outputs.to("cpu").argmax(dim=-1)
                origs = labels.to("cpu")
                batch_scores = self._compute_scores_(origs, preds)
                for k, v in batch_scores.items():
                    scores[k] += v
                # total_loss += batch_size * criterion(
                #     output_flat, targets).item()

        metrics = self._compute_metrics_(scores)

        # Set outputs
        predictions = np.array(predictions)
        total_loss = np.sum(total_loss)
        result = {**scores, **metrics}

        self.to("cpu")
        self.embeddings.to("cpu")

        return predictions, total_loss, result

    def _compute_scores_(self, orig, pred):
        scores = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        # Inputs to boolean
        pred = torch.tensor(pred.tolist()).bool()
        orig = torch.tensor(orig.tolist()).bool()

        # TP, FP, TN, FN
        TP = (pred & orig).sum().float().item()
        FP = (pred & ~orig).sum().float().item()
        TN = (~pred & ~orig).sum().float().item()
        FN = (~pred & orig).sum().float().item()

        scores["tp"] = TP
        scores["fp"] = FP
        scores["tn"] = TN
        scores["fn"] = FN

        return scores

    def _compute_metrics_(self, scores):
        """
        Computes precision, recall and f1 scores
        """
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0}
        eps = 1e-12

        TP = scores["tp"]
        FP = scores["fp"]
        TN = scores["tn"]
        FN = scores["fn"]

        # Compute metrics
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["accuracy"] = accuracy

        return metrics

    def save(self, save_path: Path):
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(exist_ok=True)
        torch.save(self.state_dict(), save_path)

    def load(self, load_path: Path):
        load_path = Path(load_path)
        self.load_state_dict(torch.load(load_path))

    def freeze_encoder_layer(self):
        for param in self.encoderTransform.parameters():
            # print(param.requires_grad)
            param.requires_grad = False

    def unfreeze_encoder_layer(self):
        for param in self.encoderTransform.parameters():
            # print(param.requires_grad)
            param.requires_grad = True
