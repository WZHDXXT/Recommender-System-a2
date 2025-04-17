import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, max_seq_length=100, num_layers=4, num_heads=4, dropout=0.2):
        super(BERT4Rec, self).__init__()

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size  # Set to 768 to match BERT's hidden size

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        for param in self.bert.encoder.layer[:2].parameters():
            param.requires_grad = False

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, masked_positions):
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        # attention mask for padding tokens
        attention_mask = input_ids.ne(0).long()  
        token_type_ids = torch.zeros_like(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # final hidden representation
        sequence_output = outputs.last_hidden_state  
        
        # gather the hidden representation of masked_positions
        masked_output = self._gather_positions(sequence_output, masked_positions)

        # prediction over vocab
        logits = self.output_layer(masked_output)  # [batch, num_masked, vocab_size]

        return logits

    def _gather_positions(self, sequence_output, positions):

        # number of user and mask position
        batch_size, num_pos = positions.size()
        hidden_size = sequence_output.size(-1)

        # flatten index of masks
        flat_offsets = torch.arange(batch_size, device=positions.device) * sequence_output.size(1)
        flat_positions = (positions + flat_offsets.unsqueeze(1)).view(-1).long()

        # extract real words from the flat_positions of masks
        flat_seq_output = sequence_output.contiguous().view(-1, hidden_size)
        selected = flat_seq_output[flat_positions] 
        # back to the original shape of masks
        return selected.view(batch_size, num_pos, hidden_size)