from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import webdataset as wds
from torch.utils.data import Dataset, IterableDataset

IGNORE_INDEX = -100


class ProteinIndexDataset(Dataset):
    def __init__(self, index_path: str | Path, tokenizer, max_samples: int = 0):
        self.index_path = Path(index_path)
        self.tokenizer = tokenizer
        self.records = []
        with self.index_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def _encode_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(f"1{sequence}2", add_special_tokens=False, return_attention_mask=True)
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        payload = torch.load(record["embedding_path"], map_location="cpu")
        structure_embeddings = payload["h_V_last_layer"].float()
        sample = self._encode_sequence(record["sequence"])
        sample.update(
            {
                "structure_embeddings": structure_embeddings,
                "structure_attention_mask": torch.ones(structure_embeddings.shape[0], dtype=torch.bool),
                "sample_id": record["id"],
                "sequence": record["sequence"],
            }
        )
        return sample


class ProteinWdsDataset(IterableDataset):
    def __init__(self, shard_pattern: str, tokenizer, shuffle: bool = False, max_samples: int = 0):
        self.shard_pattern = shard_pattern
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.max_samples = max_samples

    def _encode_sequence(self, sequence: str) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer(f"1{sequence}2", add_special_tokens=False, return_attention_mask=True)
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        labels[0] = IGNORE_INDEX
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _iter_dataset(self) -> Iterable[Dict[str, torch.Tensor]]:
        dataset = wds.WebDataset(self.shard_pattern, shardshuffle=self.shuffle)
        if self.shuffle:
            dataset = dataset.shuffle(1000)
        seen = 0
        for sample in dataset:
            record = json.loads(sample["json"].decode("utf-8"))
            payload = torch.load(io.BytesIO(sample["pth"]), map_location="cpu")
            structure_embeddings = payload["h_V_last_layer"].float()
            item = self._encode_sequence(record["sequence"])
            item.update(
                {
                    "structure_embeddings": structure_embeddings,
                    "structure_attention_mask": torch.ones(structure_embeddings.shape[0], dtype=torch.bool),
                    "sample_id": record["id"],
                    "sequence": record["sequence"],
                }
            )
            yield item
            seen += 1
            if self.max_samples > 0 and seen >= self.max_samples:
                break

    def __iter__(self):
        return iter(self._iter_dataset())


class ProteinDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        structure_embeddings = [instance["structure_embeddings"] for instance in instances]
        structure_attention_mask = [instance["structure_attention_mask"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        max_struct_len = max(x.shape[0] for x in structure_embeddings)
        hidden_dim = structure_embeddings[0].shape[1]
        padded_struct = structure_embeddings[0].new_zeros((len(instances), max_struct_len, hidden_dim))
        padded_struct_mask = torch.zeros((len(instances), max_struct_len), dtype=torch.bool)
        for idx, (emb, mask) in enumerate(zip(structure_embeddings, structure_attention_mask)):
            padded_struct[idx, : emb.shape[0]] = emb
            padded_struct_mask[idx, : mask.shape[0]] = mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "structure_embeddings": padded_struct,
            "structure_attention_mask": padded_struct_mask,
            "sample_ids": [instance["sample_id"] for instance in instances],
            "sequences": [instance["sequence"] for instance in instances],
        }
