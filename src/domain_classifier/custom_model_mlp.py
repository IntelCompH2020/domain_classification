import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# import torch.utils.data as Dataset
from torch.utils.data import Dataset

# import torchvision.transforms as transforms
# import torchvision.datasets as datasets


class CustomDatasetMLP(Dataset):

    def __init__(self, df_data):
        super(CustomDatasetMLP, self).__init__()
        self.data = df_data[['embeddings', 'labels']].to_numpy().copy()
        # print(f'self.data:{self.data}')

    def __getitem__(self, idx):

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        item = (torch.Tensor(self.data[idx, 0].copy()).to(device),
                torch.Tensor([self.data[idx, 1]]).to(device))
        return item

    def __len__(self):
        return len(self.data)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        if torch.cuda.is_available():
            self.to('cuda')
        else:
            self.to('cpu')

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x

    def save(self):
        pass

    def calculate_f1_score(self, y_preds, y_labels):
        epsilon = 1e-7
        y_preds = y_preds.view(-1)
        y_labels = y_labels.view(-1)

        tp_c = (y_preds * (y_labels == 1)).sum()
        fp_c = (y_preds * (y_labels == 0)).sum()
        tn_c = ((1 - y_preds) * (y_labels == 1)).sum()
        fn_c = ((1 - y_preds) * (y_labels == 0)).sum()

        print(f'tp_c/fp_c/tn_c/tp_c:{tp_c}/{fp_c}/{tn_c}/{fn_c}')

        precision = tp_c / (tp_c + fp_c + epsilon)
        recall = tp_c / (tp_c + fn_c + epsilon)

        return 2 * precision * recall / (precision + recall)

    def predict_proba(self, df_eval):

        eval_data = CustomDatasetMLP(df_eval)
        eval_iterator = data.DataLoader(
            eval_data, shuffle=False, batch_size=8)
        predictions = []
        for (x, y) in tqdm(eval_iterator, desc="Inference", leave=False):
            predictions_new = self(x).detach().cpu().numpy()
            if len(predictions) == 0:
                predictions = predictions_new
            else:
                predictions = np.concatenate([predictions, predictions_new])

        return predictions.reshape(-1)

    def train_loop(self, train_iterator, eval_iterator, epochs=-1,
                   device='cuda'):

        train_loss = 0
        eval_loss = 0
        eval_losses = []
        train_losses = []
        early_stopping = epochs == -1
        y_labels, y_preds, f1_scores = [], [], []

        while True:
            self.train()

            for (x, y) in tqdm(train_iterator, desc="Training", leave=False):
                # ∫print(f'loopX/Y:{x}/{y}')
                self.optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.eval()
            with torch.no_grad():
                for (x, y) in tqdm(eval_iterator, desc="Eval", leave=False):
                    y_pred = self.forward(x)
                    loss = self.criterion(y_pred, y)
                    eval_loss += loss.item()
                    if len(y_preds) == 0:
                        y_preds = y_pred
                        y_labels = y
                    else:
                        y_preds = torch.concat([y_preds, y_pred])
                        y_labels = torch.concat([y_labels, y])

            f1_scores.append(self.calculate_f1_score(
                y_preds, y_labels).detach().cpu().numpy())
            y_labels, y_preds = [], []

            train_losses.append(train_loss / len(train_iterator))
            eval_losses.append(eval_loss / len(eval_iterator))

            print(f"Train / eval loss / eval_f1_score: {train_losses[-1]:.5f}"
                  f" / {eval_losses[-1]:.5f} / {f1_scores[-1]:.5f}")

            if len(train_losses) == epochs:
                break

            if early_stopping:
                metrics = -np.array(f1_scores)  # eval_scores
                # metrics = np.array(eval_losses)
                metrics[-3:] *= 0.99  # to forece a minimum improvement
                if np.argmin(metrics) + 1 == len(metrics):
                    best_model = self

                if np.argmin(metrics) + 3 <= len(metrics):
                    self = best_model
                    break

        return [train_losses[-1], eval_losses[-1], f1_scores[-1]]

    def eval_model(self, eval_iterator, device="cuda", batch_size=8):
        """
        Evaluate trained model

        Parameters
        ----------
        df_train : DataFrame
            Training dataframe
        epochs : int
            Number of epochs to train model
        device : str, optional (default="cuda")
            If "cuda", a GPU is used if available
        batch_size : int, optiona (default=8)
            Batch size
        """

        #inference_data = CustomDatasetMLP(df_inference)
        #inference_iterator = data.DataLoader(
        #    inference_data, shuffle=False, batch_size=8)
        #predictions = []
#
        #for (x, y) in tqdm(inference_iterator, desc="Inference", leave=False):
        #    predictions_new = self.model(x).detach().cpu().numpy().reshape(-1)
        #    if len(predictions) == 0:
        #        predictions = predictions_new
        #    else:
        #        predictions = np.concatenate([predictions, predictions_new])

        # Convert DataFrame to DataLoader

        # turn on evaluation mode
        self.eval()
        predictions = []
        total_loss = []
        scores = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        with torch.no_grad():
            for (x, y) in tqdm(eval_iterator, desc="Eval batch", leave=False):
                y_pred = self(x)
                predictions.extend(y_pred.to("cpu").tolist())
                batch_scores = self._compute_scores_(y, y_pred)
                for k, v in batch_scores.items():
                    scores[k] += v

        metrics = self._compute_metrics_(scores)

        # Set outputs
        predictions = np.array(predictions)
        total_loss = np.sum(total_loss)
        result = {**scores, **metrics}

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
