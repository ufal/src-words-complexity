import os
import sys

import numpy as np
import torch.utils.data as torch_data
import torch
import gzip
from pprint import pprint

import numpy as np

class SrcComplexityDataset:

    class Batch:
        def __init__(self, input_ids, token_type_ids, attention_mask, labels):
            self.input_ids = torch.tensor(input_ids)
            self.token_type_ids = torch.tensor(token_type_ids)
            self.attention_mask = torch.tensor(attention_mask)
            self.labels = torch.tensor(labels)

        def to(self, device):
            # print(device.type, file=sys.stderr)
            self.input_ids = self.input_ids.to(device)
            self.token_type_ids = self.token_type_ids.to(device)
            self.attention_mask = self.attention_mask.to(device)
            self.labels = self.labels.to(device)
            # print("input_ids is CUDA? {}".format(self.input_ids.is_cuda), file=sys.stderr)
            # print("token_type_ids is CUDA? {}".format(self.token_type_ids.is_cuda), file=sys.stderr)
            # print("attention_mask is CUDA? {}".format(self.attention_mask.is_cuda), file=sys.stderr)
            # print("labels is CUDA? {}".format(self.labels.is_cuda), file=sys.stderr)

    class Dataset(torch_data.Dataset):
        WORDS_SUFFIX = ".src.limit.txt.gz"
        LABELS_SUFFIX = ".labels.limit.txt.gz"

        CLS = "[CLS]"
        SEP = "[SEP]"

        def __init__(self, path_prefix, tokenizer=None, max_length=128, max_sentences=None, shuffle_batches=True):
            self.tokenizer = tokenizer

            # process tokens
            self.wp_to_tokens_map = []
            self.wpids = []
            for sent_i, sent_words in enumerate(self.read_file(path_prefix + self.WORDS_SUFFIX, max_sentences=max_sentences)):

                sent_wp_to_tokens = []
                sent_wps = []

                tokens_wp = [tokenizer.tokenize(w) for w in sent_words]
                for token_i, wps in enumerate(tokens_wp):
                    sent_wp_to_tokens.extend([token_i]*len(wps))
                    sent_wps.extend(wps)
                self.wp_to_tokens_map.append(np.array(sent_wp_to_tokens))
                sent_wpids = [tokenizer.convert_tokens_to_ids(wp) for wp in sent_wps]
                self.wpids.append(np.array(sent_wpids))

                if not (sent_i % 10000):
                    print("Loading tokens from sentence number {}".format(sent_i), file=sys.stderr)

            # process token labels
            self.wplabels = []
            for sent_i, sent_labels in enumerate(self.read_file(path_prefix + self.LABELS_SUFFIX, max_sentences=max_sentences)):
                sent_labels_arr = np.array(sent_labels, dtype=int)
                self.wplabels.append(sent_labels_arr[self.wp_to_tokens_map[sent_i]])
                if not (sent_i % 10000):
                    print("Loading labels from sentence number {}".format(sent_i), file=sys.stderr)

            if len(self.wpids) != len(self.wplabels):
                raise ValueError("The words file must contain the same number of sentences (lines) as the labels file: {} vs. {}".format(
                    len(self.wpids),
                    len(self.wplabels)
                ))

            # add ids of special words as attributes
            for w in ["CLS", "SEP"]:
                setattr(self, w, tokenizer.convert_tokens_to_ids("[" + w + "]"))

            self.shuffle_batches = shuffle_batches
            self.max_length = max_length

            # self.data_loader = None

        def read_file(self, path, max_sentences=None):
            with gzip.open(path, "rt", encoding="utf-8") as in_file:
                for sent_i, seq in enumerate(in_file):
                    if max_sentences is not None and sent_i+1 > max_sentences:
                        break
                    seq = seq.rstrip("\n")
                    seq_list = seq.split(" ")
                    yield seq_list

        def __len__(self):
            return len(self.wpids)

        def __getitem__(self, idx):
            return {"words": self.words[idx], "labels": self.labels[idx]}

        def batches(self, size=None):
            # if self.data_loader is None:
            #     self.data_loader = torch_data.DataLoader(self, batch_size=size, shuffle=self.shuffle_batches)
            # for sample in self.data_loader:
            #    yield sample

            sampler_name = torch_data.RandomSampler if self.shuffle_batches else torch_data.SequentialSampler
            sampler = torch_data.BatchSampler(sampler_name(range(len(self))), size, drop_last=False)

            for batch_idx in sampler:
                # let wpid_batch and wplabel_batch contain the list of ids and labels, respectively
                wpid_batch = [self.wpids[idx] for idx in batch_idx]
                wplabel_batch = [self.wplabels[idx] for idx in batch_idx]

                # transform a list of sampled id sequences to a batch matrix of a fixed size by trimming and padding
                # self.max_length-2: leave space for [SEP] and [CLS]
                input_ids = np.zeros((size, self.max_length-2), dtype=int)
                for i, wpid in enumerate(wpid_batch):
                    l = min([len(wpid), self.max_length-2])
                    input_ids[i, :l] = wpid[:l]

                # add [SEP] at the end of each sentence (just before padding)
                flipped_input_ids = self._flip_nonzero_columns(input_ids)
                flipped_input_ids = np.c_[np.tile(self.SEP, input_ids.shape[0]), flipped_input_ids]
                input_ids = self._flip_nonzero_columns(flipped_input_ids)

                # add [CLS] at the beginning of each sentence
                input_ids = np.c_[np.tile(self.CLS, input_ids.shape[0]), input_ids]

                # all tokens belong to the same sentence => all zeros
                token_type_ids = np.zeros((size, self.max_length), dtype=int)

                # highlight padding (zeros)
                attention_mask = (input_ids != 0).astype(int)

                # transform a list of sampled labels to a batch matrix of a fixed size by trimming and padding
                # input_ids[i, 1:l+1] : the 0th item corresponds to the [CLS] token - use 0 label for it
                labels = np.zeros((size, self.max_length), dtype=int)
                for i, wplabel in enumerate(wplabel_batch):
                    l = min([len(wplabel), self.max_length-2])
                    labels[i, 1:l+1] = wplabel[:l]

                yield SrcComplexityDataset.Batch(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

        def _flip_nonzero_columns(self, matrix):
            rows, column_indices = np.ogrid[:matrix.shape[0], :matrix.shape[1]]
            first_zero_idx = np.count_nonzero(matrix, 1) - matrix.shape[1]
            flipped_matrix = np.flip(matrix, 1)
            first_zero_idx[first_zero_idx < 0] += matrix.shape[1]
            column_indices = column_indices - first_zero_idx[:, np.newaxis]
            return flipped_matrix[rows, column_indices]

    CLASSES = [0, 1]
    CLASS_WEIGHTS = [0.05, 0.95]

    @property
    def classes(self):
        return self.CLASSES

    @property
    def class_weights(self):
        return self.CLASS_WEIGHTS

    def __init__(self, tokenizer=None, max_length=64, max_sentences=None, **kwargs):
        for dataset in ["train", "dev", "test"]:
            dataset_prefix = kwargs.get(dataset)
            if dataset_prefix is not None:
                print("Loading {} dataset...".format(dataset))
                setattr(self, dataset, self.Dataset(dataset_prefix,
                                                    tokenizer=tokenizer,
                                                    # train=self.train if dataset != "train" else None,
                                                    max_length=max_length,
                                                    max_sentences=max_sentences if max_sentences is None or dataset == "train" else int(max_sentences/10),
                                                    # shuffle_batches=True))
                                                    shuffle_batches=dataset == "train"))

