import argparse
import sys
import torch
import numpy as np
import os
import re
import datetime
import logging
from torch.optim import Adam
from pytorch_transformers import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tensorboardX import SummaryWriter

from dataset import SrcComplexityDataset
from classifiers import BertForWeighedTokenClassification

argparser = argparse.ArgumentParser()
argparser.add_argument("train", type=str, help="A path prefix to train data.")
argparser.add_argument("dev", type=str, help="A path prefix to dev data.")
argparser.add_argument("--save-models", action="store_true", help="If enabled, stores model to the $logdir/models directory.")
argparser.add_argument("--max-sentences", type=int, default=None, help="A maximum number of sentences to be loaded per dataset.")
argparser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
argparser.add_argument("--batch-size", type=int, default=32, help="Batch size")
argparser.add_argument("--no-fine-tune", action="store_true", help="Use the representation returned by the transformer as features.")
argparser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
argparser.add_argument("--weight-positive", default=0.5, type=float, help="A weight of positive examples in cross-entropy loss (must be in [0,1]).")
args = argparser.parse_args()

logdir_ignore_args = [ "train", "dev", "save_models" ]

# Create logdir name
args.logdir = os.path.join("runs", "{}-{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if key not in logdir_ignore_args))
))
os.mkdir(args.logdir)
logging.basicConfig(filename=os.path.join(args.logdir, "run.log"), level=logging.DEBUG, format='%(asctime)s %(message)s')

# Prepare directory for saving models
if args.save_models:
    args.modeldir = os.path.join(args.logdir, "models")
    os.mkdir(args.modeldir)

torch.manual_seed(1986)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data = SrcComplexityDataset(train=args.train, dev=args.dev, tokenizer=tokenizer, max_sentences=args.max_sentences)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    logging.info("Device: GPU {} (out of {} GPUs)".format(gpu_name, str(n_gpu)))
else:
    logging.info("Device: CPU")

#  create model
model = BertForWeighedTokenClassification.from_pretrained("bert-base-uncased",
                                                          num_labels=len(data.classes),
                                                          label_weights=[1-args.weight_positive, args.weight_positive])
model.to(device)

# prepare optimizer
if not args.no_fine_tune:
    logging.info("Fine-tuning the whole net.")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    logging.info("Using BERT as features.")
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

tb_writer = SummaryWriter(logdir=args.logdir)

# Store our loss and accuracy for plotting
train_loss_set = []
global_step = 0

model.zero_grad()

for epoch in range(args.epochs):

    logging.info("Running training epoch no. {}".format(epoch+1))

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    train_loss = 0
    # nb_tr_examples = 0
    train_batch_steps = 0

    with tqdm(total=len(data.train)) as progbar:
        for step, batch in enumerate(data.train.batches(size=args.batch_size)):
            # Add batch to GPU
            batch.to(device)
            # Forward pass
            loss, scores = model(batch.input_ids,
                                 token_type_ids=batch.token_type_ids,
                                 attention_mask=batch.attention_mask,
                                 labels=batch.labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            train_loss += loss.item()
            # nb_tr_examples += b_input_ids.size(0)
            train_batch_steps += 1

            # Clear out the gradients (by default they accumulate)
            # optimizer.zero_grad()
            model.zero_grad()

            global_step += 1

            tb_writer.add_scalar("train_loss", train_loss / train_batch_steps, global_step)

            progbar.update(args.batch_size)

    logging.info("Train loss: {}".format(train_loss / train_batch_steps))

    modeldir = os.path.join(args.modeldir, "epoch_{}".format(epoch+1))
    os.mkdir(modeldir)
    logging.info("Saving model to {}".format(modeldir))
    model.save_pretrained(modeldir)

    # VALIDATION on validation set
    model.eval()
    dev_loss = 0
    dev_batch_steps = 0
    all_pred_labels, all_true_labels = [], []
    with tqdm(total=len(data.dev)) as progbar:
        for step, batch in enumerate(data.dev.batches(size=args.batch_size)):
            batch.to(device)

            with torch.no_grad():
                loss, scores = model(batch.input_ids,
                                     token_type_ids=None,
                                     attention_mask=batch.attention_mask,
                                     labels=batch.labels)
            true_labels = batch.labels.to('cpu').numpy()
            pred_labels = np.argmax(scores.detach().cpu().numpy(), axis=2)
            attention_mask = batch.attention_mask.to('cpu').numpy()
            for i in range(len(true_labels)):
                mask = attention_mask[i].astype(bool)
                all_true_labels.extend(list(true_labels[i, mask]))
                all_pred_labels.extend(list(pred_labels[i, mask]))
            # print("True labels: {}".format(len(all_true_labels)))
            # print("Predictions: {}".format(len(all_pred_labels)))

            dev_loss += loss.item()
            dev_batch_steps += 1

            progbar.update(args.batch_size)

    dev_loss = dev_loss / dev_batch_steps
    logging.info("Validation loss: {}".format(dev_loss))
    dev_acc = accuracy_score(all_true_labels, all_pred_labels)
    logging.info("Validation Accuracy: {}".format(dev_acc))
    dev_f1 = f1_score(all_true_labels, all_pred_labels)
    dev_prec = precision_score(all_true_labels, all_pred_labels)
    dev_rec = recall_score(all_true_labels, all_pred_labels)
    logging.info("F1-Score: {}".format(dev_f1))
    conf_cats = confusion_matrix(all_true_labels, all_pred_labels).ravel()
    logging.info("TN: {}, FP: {}, FN: {}, TP: {}".format(*conf_cats))

    tb_writer.add_scalar("dev_loss", dev_loss, epoch)
    tb_writer.add_scalar("dev_acc", dev_acc, epoch)
    tb_writer.add_scalar("dev_f1", dev_f1, epoch)
    tb_writer.add_scalar("dev_prec", dev_prec, epoch)
    tb_writer.add_scalar("dev_rec", dev_rec, epoch)

tb_writer.close()
