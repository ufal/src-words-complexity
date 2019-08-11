import sys
import torch
from pytorch_transformers import BertForTokenClassification
from torch.nn import CrossEntropyLoss


class BertForWeighedTokenClassification(BertForTokenClassification):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.label_weights = torch.Tensor(kwargs.get("label_weights", [0.5, 0.5]))
        print("Using cross entropy loss with following weights: {}".format(self.label_weights.cpu().numpy()), file=sys.stderr)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(self.label_weights)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

    def to(self, device):
        super().to(device)
        # print("Sending label weights to {}".format(device.type), file=sys.stderr) 
        self.label_weights = self.label_weights.to(device) 
