import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch.nn import CrossEntropyLoss
from utils.bert_model import BertModel,BertEncoder,BertPooler
from collections import namedtuple

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    
        beta = torch.softmax(w, dim=0)                
        beta = beta.expand((z.shape[0],) + beta.shape) 

        return (beta * z).sum(1)                       

class SPPGATLayer(nn.Module):
    def __init__(self, config):
        super(SPPGATLayer,self).__init__()
        self.attention_head = config ['num_attention_heads']
        self.output_size = config['output_size']
        self.num_meta_paths = 3
        self.gat_layers = nn.ModuleList()
        for i in range(self.num_meta_paths):
            self.gat_layers.append(GATConv(768, self.output_size, self.attention_head, config ['dropout'], config ['dropout'], activation=F.elu, allow_zero_in_degree=True))
        self.sound_attention = SemanticAttention(in_size=self.output_size * self.attention_head)
        self.aggregate_attention = SemanticAttention(in_size=self.output_size * self.attention_head)
        
        

    def forward(self, graphs, features, character_list):
        semantic_embeddings = []
        h1 = self.gat_layers[0](graphs[0], (features[0], features[0])) # Node-level attention of phonetic graphs
        h = self.gat_layers[2](graphs[2], (features[2], features[2]))  # Node-level attention of pronunciation graphs
        semantic_embeddings.append(h1[character_list[0]].flatten(1))   # Add the node-level phonetic embedding of characters
        semantic_embeddings.append(h[character_list[2]].flatten(1))   # Add the node-level semantic embedding of characters
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        semantic_embeddings = [self.sound_attention(semantic_embeddings)]  # Sound attention 

        h2 = self.gat_layers[1](graphs[1], (features[1], features[1])) # Node-level attention of semantic graphs
        semantic_embeddings.append(h2[character_list[1]].flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        semantic_embedding = self.aggregate_attention(semantic_embeddings).view((-1,self.attention_head,self.output_size))    # Aggregate attention layer
        h1[character_list[0]] = semantic_embedding
        h2[character_list[1]] = semantic_embedding
        h[character_list[2]] = semantic_embedding
        return semantic_embedding,[h1.mean(1).squeeze(1),h2.mean(1).squeeze(1),h.mean(1).squeeze(1)]

class classifier(nn.Module):
    def __init__(self, config, num_labels):
        super(classifier,self).__init__()
        self.name_length = config['max_length']
        self.num_pron = 400
        self.num_char = 6763
        self.num_labels = num_labels
        bert_config = config['transformer_config']
        self.bert_config = namedtuple("configTuple", bert_config.keys())(*bert_config.values())
        self.listchar = nn.Parameter(torch.tensor(range(self.num_char)), requires_grad=False)
        self.emptytensor = nn.Parameter(torch.tensor([[0]*768]), requires_grad=False)
        # The character, semantic, phonetic component encoder of characters 
        self.all_feature_embeddings = nn.Embedding(6763+1443+916,768)
        # The semantic position encoder of characters 
        self.mean_position_embeddings = nn.Embedding(40,768)
        # The BERT text encoder
        self.bert_model = BertModel.from_pretrained(config['chinese_bert_path'])
        # The SPPGAT layers
        self.SPPGAT_layers = nn.ModuleList()
        for i in range(3):
            self.SPPGAT_layers.append(SPPGATLayer(config))
        self.position_embeddings = nn.Embedding(config['max_length'],config['output_size'])
        # The transformer layer
        self.transformer = BertEncoder(self.bert_config)
        self.pooler1 = BertPooler(self.bert_config)
        self.dropout = nn.Dropout(config ['dropout'])
        self.classifier = nn.Linear(config['output_size']*2, num_labels)
        self.loss = CrossEntropyLoss()
        
    def forward(self, input_ids, labels = None):
        # Feed into the BERT layer.
        sequence_output = self.bert_model(input_ids[0])[-2][-1]    

        # Feed into the SPPGAT layer.  
        SPPGAT_embeddings = []
        stack_h = [self.all_feature_embeddings(input_ids[2]), \
            self.all_feature_embeddings(input_ids[4]) + self.mean_position_embeddings(input_ids[5]), \
            self.all_feature_embeddings(self.listchar)]
        gs = [input_ids[6],input_ids[7],input_ids[8]]
        characterlist = [input_ids[9],input_ids[10],input_ids[11]]
        for SPPGAT in self.SPPGAT_layers:
            result,stack_h = SPPGAT(gs,stack_h,characterlist)
        result = result.mean(1).squeeze(1)
        temp_embeddings = []
        for i in input_ids[1]:
            if (i != -1):
                temp_embeddings.append(result[i])
            else:
                temp_embeddings.append(self.emptytensor[0])
        SPPGAT_embeddings = torch.stack(temp_embeddings, dim=0).view(sequence_output.shape[0],-1,sequence_output.shape[-1])
        
        # Add position embedding 
        position_ids = torch.arange(self.name_length, dtype=torch.long, device= SPPGAT_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[0])
        position_embeddings = self.position_embeddings(position_ids)
        SPPGAT_embeddings = position_embeddings  + SPPGAT_embeddings 

        # Feed into transformer layer
        sequence_output = torch.concat([sequence_output, SPPGAT_embeddings],dim= -1)
        attention_mask = torch.ones_like(input_ids[0])
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
        encoded_layers = self.transformer(sequence_output, extended_attention_mask, output_all_encoded_layers=True)[-1]
        pooled_output = self.pooler1(encoded_layers)
        features_output = self.dropout(pooled_output)

        # Task-specific
        logits = self.classifier(torch.reshape(features_output,(input_ids[0].shape[0],-1)))
        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            logits = torch.argmax(logits, dim=1)
            return loss, logits
        else:
            logits = torch.argmax(logits, dim=1)
            return logits