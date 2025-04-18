from transformers import BertPreTrainedModel, BertModel, BertConfig
import torch.nn as nn
import torch

class Bert4RecConfig(BertConfig):
    def __init__(self, max_seq_length=20, vocab_size=3706, 
                 bert_num_blocks=2, bert_num_heads=2, 
                 hidden_size=512, bert_dropout=0.1, **kwargs):
        super().__init__(
            vocab_size=vocab_size + 2, 
            hidden_size=hidden_size,
            num_hidden_layers=bert_num_blocks,
            num_attention_heads=bert_num_heads,
            intermediate_size=hidden_size * 4,
            hidden_act="gelu",
            max_position_embeddings=max_seq_length,
            pad_token_id=0,
            type_vocab_size=1,  
            attention_probs_dropout_prob=bert_dropout,
            hidden_dropout_prob=bert_dropout,
            **kwargs
        )

class Bert4Rec(BertPreTrainedModel):
    config_class = Bert4RecConfig

    def __init__(self, max_seq_length, vocab_size, bert_num_blocks, 
                 bert_num_heads, hidden_size, bert_dropout):
        config = Bert4RecConfig(
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            bert_num_blocks=bert_num_blocks,
            bert_num_heads=bert_num_heads,
            hidden_size=hidden_size,
            bert_dropout=bert_dropout
        )
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        
        self.out = nn.Linear(config.hidden_size, vocab_size + 1)
        self.init_weights()

    def forward(self, x):
        attention_mask = (x != 0).float().to(x.device)
        token_type_ids = torch.zeros_like(x, dtype=torch.long, device=x.device)
        
        outputs = self.bert(
            input_ids=x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        return self.out(outputs.last_hidden_state)

