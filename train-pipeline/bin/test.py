import argparse
import torch
import numpy as np
import logging
from pytorch_transformers import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from dataset import SrcComplexityDataset
from classifiers import BertForWeighedTokenClassification

argparser = argparse.ArgumentParser()
argparser.add_argument("model", type=str, help="A path to the model directory.")
argparser.add_argument("test", type=str, help="A path prefix to test data.")
argparser.add_argument("--batch-size", type=int, default=32, help="Batch size")
argparser.add_argument("--max-sentences", type=int, default=None, help="A maximum number of sentences to be loaded per dataset.")
args = argparser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

torch.manual_seed(1986)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data = SrcComplexityDataset(test=args.test, tokenizer=tokenizer, max_sentences=args.max_sentences)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    n_gpu = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    logging.info("Device: GPU {} (out of {} GPUs)".format(gpu_name, str(n_gpu)))
else:
    logging.info("Device: CPU")

#  create model
logging.info("Loading model from {}".format(args.model))
model = BertForWeighedTokenClassification.from_pretrained(args.model)
model.to(device)
model.eval()
logging.info("Model loaded")

test_loss = 0
test_batch_steps = 0
all_pred_labels, all_true_labels = [], []
with tqdm(total=len(data.test)) as progbar:
    for step, batch in enumerate(data.test.batches(size=args.batch_size)):
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

        test_loss += loss.item()
        test_batch_steps += 1

        progbar.update(args.batch_size)

test_loss = test_loss / test_batch_steps
print("Loss: {}".format(test_loss))
test_acc = accuracy_score(all_true_labels, all_pred_labels)
print("Accuracy: {}".format(test_acc))
test_f1 = f1_score(all_true_labels, all_pred_labels)
test_prec = precision_score(all_true_labels, all_pred_labels)
test_rec = recall_score(all_true_labels, all_pred_labels)
print("Precision: {}".format(test_prec))
print("Recall: {}".format(test_rec))
print("F1-Score: {}".format(test_f1))
conf_cats = confusion_matrix(all_true_labels, all_pred_labels).ravel()
print("TN: {}, FP: {}, FN: {}, TP: {}".format(*conf_cats))
