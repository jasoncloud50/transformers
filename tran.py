import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import copy
import math
import json
import time

torch.manual_seed(40)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def get_transformer_scheduler(optimizer, embedding_dimension=512, warmup_steps=10000, factor=1.0):
    def lr_lambda(step):
        if step == 0:
            step = 1
        return factor * (embedding_dimension ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class mydataset(Dataset):
    def __init__(self, text_idx, block_size=256):
        self.text_idx = text_idx 
        self.block_size = block_size

    def __getitem__(self, index):
        return self.text_idx[index:index+self.block_size], self.text_idx[index+1:index+self.block_size+1]

    def __len__(self):
        return len(self.text_idx) - self.block_size

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimension=512, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dimension)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dimension, 2).float() * (-math.log(10000.0) / embedding_dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print(f"pe shape after unsqueeze: {pe.shape}")
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x).contiguous()

class one_head_attention(nn.Module):
    def __init__(self, embedding_dimension=512, attention_head_dimension=64):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.attention_head_dimension = attention_head_dimension
        self.linear_layer_query = nn.Linear(self.embedding_dimension, self.attention_head_dimension, bias=False)
        self.linear_layer_key = nn.Linear(self.embedding_dimension, self.attention_head_dimension, bias=False)
        self.linear_layer_value = nn.Linear(self.embedding_dimension, self.attention_head_dimension, bias=False)

    def forward(self, x):
        Q = self.linear_layer_query(x)
        K = self.linear_layer_key(x)
        V = self.linear_layer_value(x)
        scores = Q @ K.transpose(1, 2) / math.sqrt(self.attention_head_dimension)
        attn_weights = F.softmax(scores, dim=2)
        return attn_weights @ V

class one_head_cross_attention(one_head_attention):
    def __init__(self, embedding_dimension=512, attention_head_dimension=64):
        super().__init__(embedding_dimension, attention_head_dimension)

    def forward(self, encoder_input, decoder_input):
        Q = self.linear_layer_query(decoder_input)
        K = self.linear_layer_key(encoder_input)
        V = self.linear_layer_value(encoder_input)
        scores = Q @ K.transpose(1, 2) / math.sqrt(self.attention_head_dimension)
        attn_weights = F.softmax(scores, dim=2)
        return attn_weights @ V

class masked_one_head_attention(one_head_attention):
    def __init__(self, embedding_dimension=512, attention_head_dimension=64, max_len=5000):
        super().__init__(embedding_dimension, attention_head_dimension)
        mask = torch.zeros(max_len, max_len)
        mask = torch.full((max_len, max_len), -torch.inf)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        Q = self.linear_layer_query(x)
        K = self.linear_layer_key(x)
        V = self.linear_layer_value(x)
        scores = Q @ K.transpose(1, 2) / math.sqrt(self.attention_head_dimension)
        mask = self.mask[0:x.size(1), 0:x.size(1)].contiguous()
        scores = scores + mask
        attn_weights = F.softmax(scores, dim=2)
        return attn_weights @ V

class multi_head_attention_layer(nn.Module):
    def __init__(self, embedding_dimension=512, num_of_attention_head=8, attention_head_dimension=64):
        super().__init__()
        self.heads = nn.ModuleList() 
        for _ in range(num_of_attention_head):
            self.heads.append(one_head_attention(embedding_dimension, attention_head_dimension))
        self.linearlayer = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        concat = torch.cat(outputs, dim=2)
        return self.linearlayer(concat.contiguous())

class cross_attention_layer(multi_head_attention_layer):
    def __init__(self, embedding_dimension=512, num_of_attention_head=8, attention_head_dimension=64):
        super().__init__(embedding_dimension, num_of_attention_head, attention_head_dimension)
        self.heads = nn.ModuleList() 
        for _ in range(num_of_attention_head):
            self.heads.append(one_head_cross_attention(embedding_dimension, attention_head_dimension))

    def forward(self, encoder_input, decoder_input):
        outputs = [head(encoder_input, decoder_input) for head in self.heads]
        concat = torch.cat(outputs, dim=2)
        return self.linearlayer(concat.contiguous())

class masked_multi_head_attention_layer(multi_head_attention_layer):
    def __init__(self, embedding_dimension=512, num_of_attention_head=8, attention_head_dimension=64, max_len=5000):
        super().__init__(embedding_dimension, num_of_attention_head, attention_head_dimension)
        self.heads = nn.ModuleList() 
        for _ in range(num_of_attention_head):
            self.heads.append(masked_one_head_attention(embedding_dimension, attention_head_dimension, max_len))

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        concat = torch.cat(outputs, dim=2)
        return self.linearlayer(concat.contiguous())

class feed_forward_neural_network(nn.Module):
    def __init__(self, inputandoutput_dimension=512, hidden_units_dimension=2048):
        super().__init__()
        self.layer1 = nn.Linear(inputandoutput_dimension, hidden_units_dimension)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_units_dimension, inputandoutput_dimension)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

class resnet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, sublayer_input):
        return input + sublayer_input

class add_and_norm(nn.Module):
    def __init__(self, embedding_dimension=512):
        super().__init__()
        self.layer1 = resnet()
        self.layer2 = nn.LayerNorm(embedding_dimension, elementwise_affine=False, bias=False)

    def forward(self, input, sublayer_input):
        return self.layer2(self.layer1(input, sublayer_input))

class encoder_layer(nn.Module):
    def __init__(self, embedding_dimension=512, num_of_attention_head=8, attention_head_dimension=64, hidden_units_dimension=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.multi_head_attention = multi_head_attention_layer(embedding_dimension, num_of_attention_head, attention_head_dimension)
        self.feed_forward_neural_network = feed_forward_neural_network(embedding_dimension, hidden_units_dimension)
        self.add_and_norm = add_and_norm(embedding_dimension)

    def forward(self, x):
        attn_out = self.multi_head_attention(x)
        attn_out = self.dropout(attn_out)
        norm1 = self.add_and_norm(x, attn_out).contiguous()

        ffn_out = self.feed_forward_neural_network(norm1)
        ffn_out = self.dropout(ffn_out)
        result = self.add_and_norm(norm1, ffn_out).contiguous()
        return result

class decoder_layer(nn.Module):
    def __init__(self, embedding_dimension=512, num_of_attention_head=8, attention_head_dimension=64, hidden_units_dimension=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.masked_multi_head_attention = masked_multi_head_attention_layer(embedding_dimension, num_of_attention_head, attention_head_dimension, max_len)
        self.cross_attention = cross_attention_layer(embedding_dimension, num_of_attention_head, attention_head_dimension)
        self.feed_forward_neural_network = feed_forward_neural_network(embedding_dimension, hidden_units_dimension)
        self.add_and_norm = add_and_norm(embedding_dimension)

    def forward(self, encoder_input, x):
        masked_attn_out = self.masked_multi_head_attention(x)
        masked_attn_out = self.dropout(masked_attn_out)
        norm1 = self.add_and_norm(x, masked_attn_out).contiguous()

        if encoder_input is not x:
            cross_attn_out = self.cross_attention(encoder_input, norm1)
            cross_attn_out = self.dropout(cross_attn_out)
            norm2 = self.add_and_norm(norm1, cross_attn_out).contiguous()

        if encoder_input is not x:
            ffn_out = self.feed_forward_neural_network(norm2)
            ffn_out = self.dropout(ffn_out)
            result = self.add_and_norm(norm2, ffn_out).contiguous()
        else:
            ffn_out = self.feed_forward_neural_network(norm1)
            ffn_out = self.dropout(ffn_out)
            result = self.add_and_norm(norm1, ffn_out).contiguous()
        return result

class transformers_model(nn.Module):
    def __init__(self, mode, is_same_language, embedding_dimension=512, num_of_attention_head=8, num_of_words_in_encoder=0, num_of_words_in_decoder=0, hidden_units_dimension=2048, encoder_layers=6, decoder_layers=6, dropout=0.1, max_len=5000):
        super().__init__()
        assert mode == "BERT" or mode == "GPT" or mode == "Local Transformer", "mode must be one of 'BERT', 'GPT', or 'Local Transformer'"
        self.mode = mode
        self.embedding_dimension = embedding_dimension
        self.num_of_attention_head = num_of_attention_head
        assert embedding_dimension % num_of_attention_head == 0, "embedding_dimension must be divisible by num_of_attention_head"
        self.attention_head_dimension = math.floor(embedding_dimension / num_of_attention_head)
        self.num_of_words_in_encoder = num_of_words_in_encoder
        self.num_of_words_in_decoder = num_of_words_in_decoder
        self.hidden_units_dimension = hidden_units_dimension
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dropout = dropout
        self.max_len = max_len

        if mode == "Local Transformer":
            self.encoder_embedding = nn.Embedding(num_embeddings=self.num_of_words_in_encoder, embedding_dim=self.embedding_dimension)
            self.decoder_embedding = nn.Embedding(num_embeddings=self.num_of_words_in_decoder, embedding_dim=self.embedding_dimension)
            self.position_embedding = PositionalEncoding(self.embedding_dimension, self.dropout, self.max_len)

            self.encoder_layer_list = nn.ModuleList()
            for _ in range(self.encoder_layers):
                self.encoder_layer_list.append(encoder_layer(self.embedding_dimension, self.num_of_attention_head, self.attention_head_dimension, self.hidden_units_dimension, self.dropout))
            self.encoder_layer = nn.Sequential(*self.encoder_layer_list)

            self.decoder_layer_list = nn.ModuleList()
            for _ in range(self.decoder_layers):
                self.decoder_layer_list.append(decoder_layer(self.embedding_dimension, self.num_of_attention_head, self.attention_head_dimension, self.hidden_units_dimension, self.dropout, self.max_len))

            self.linear_layer = nn.Linear(self.embedding_dimension, self.num_of_words_in_decoder)

            if is_same_language:
                self.encoder_embedding.weight = self.linear_layer.weight
                self.decoder_embedding.weight = self.linear_layer.weight
            else:
                self.decoder_embedding.weight = self.linear_layer.weight

        elif mode == "GPT":
            self.decoder_embedding = nn.Embedding(num_embeddings=self.num_of_words_in_decoder, embedding_dim=self.embedding_dimension)
            self.position_embedding = PositionalEncoding(self.embedding_dimension, self.dropout, self.max_len)

            self.decoder_layer_list = nn.ModuleList()
            for _ in range(self.decoder_layers):
                self.decoder_layer_list.append(decoder_layer(self.embedding_dimension, self.num_of_attention_head, self.attention_head_dimension, self.hidden_units_dimension, self.dropout, self.max_len))

            self.linear_layer = nn.Linear(self.embedding_dimension, self.num_of_words_in_decoder)
            self.decoder_embedding.weight = self.linear_layer.weight

        else:
            self.encoder_embedding = nn.Embedding(num_embeddings=self.num_of_words_in_encoder, embedding_dim=self.embedding_dimension)
            self.position_embedding = PositionalEncoding(self.embedding_dimension, self.dropout, self.max_len)

            self.encoder_layer_list = nn.ModuleList()
            for _ in range(self.encoder_layers):
                self.encoder_layer_list.append(encoder_layer(self.embedding_dimension, self.num_of_attention_head, self.attention_head_dimension, self.hidden_units_dimension, self.dropout))
            self.encoder_layer = nn.Sequential(*self.encoder_layer_list)

            self.linear_layer = nn.Linear(self.embedding_dimension, self.num_of_words_in_encoder)
            self.encoder_embedding.weight = self.linear_layer.weight

    def forward(self, encoder_input=None, decoder_input=None):
        if self.mode == "Local Transformer":
            assert encoder_input is not None and decoder_input is not None, "Both encoder_input and decoder_input must be provided for Local Transformer"
            enc_out = self.encoder_embedding(encoder_input)
            enc_out = self.position_embedding(enc_out)
            enc_out = self.encoder_layer(enc_out)

            dec_out = self.decoder_embedding(decoder_input)
            dec_out = self.position_embedding(dec_out)
            for layer in self.decoder_layer_list:
                dec_out = layer(enc_out, dec_out)
                dec_out = dec_out.contiguous()
            logits = self.linear_layer(dec_out)
            return logits.contiguous()  

        elif self.mode == "GPT":
            assert decoder_input is not None and (encoder_input is None), "For GPT, only decoder_input should be provided, encoder_input must be None"
            dec_out = self.decoder_embedding(decoder_input)
            dec_out = self.position_embedding(dec_out)
            for layer in self.decoder_layer_list:
                dec_out = layer(dec_out, dec_out)
                dec_out = dec_out.contiguous()
            logits = self.linear_layer(dec_out)
            return logits.contiguous()  
        
        else:
            assert (decoder_input is None) and encoder_input is not None, "For BERT, only encoder_input should be provided, decoder_input must be None"
            enc_out = self.encoder_embedding(encoder_input)
            enc_out = self.position_embedding(enc_out)
            enc_out = self.encoder_layer(enc_out)
            logits = self.linear_layer(enc_out)
            return logits.contiguous()

def train_one_epoch(model, dataloader, criterion, scheduler, epoch, device, save_model_path):
    model.train()
    model = model.to(device)
    scaler = GradScaler("cuda")
    running_loss = 0.0
    running_loss_list = []
    correct = 0
    total = 0
    start_time = time.time()
    grad_norm_sum = 0.0
    grad_norm_count = 0

    for batch_idx, (data, label) in enumerate(dataloader):
        data = data.to(device)
        label = label.to(device).to(torch.long)

        scheduler.optimizer.zero_grad()

        with autocast("cuda"):
            output = model(decoder_input=data)
            output = output.contiguous()

        running_loss = criterion(input=output.view(-1, output.shape[2]), target=label.view(-1))
        running_loss_list.append(running_loss.detach())

        scaler.scale(running_loss).backward()

        scaler.unscale_(scheduler.optimizer)

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norm_sum += total_norm
        grad_norm_count += 1

        scaler.step(scheduler.optimizer)
        scaler.update()
        scheduler.step()

        batch_correct = torch.argmax(output, dim=2).eq(label).sum().item() 
        correct += batch_correct
        total += data.shape[0] * data.shape[1]
        
        if batch_idx % 100 == 0:
            print(batch_correct)

    with open(save_model_path, "wb") as f:
        torch.save(model.state_dict(), f)
    print("save the model after a epoch...")

    accuracy_in_this_epoch = correct / total
    running_loss_in_this_epoch = torch.tensor(running_loss_list).mean().item()
    avg_grad_norm = grad_norm_sum / grad_norm_count if grad_norm_count > 0 else 0.0
    end_time = time.time()
    total_tokens = len(dataloader.dataset) * dataloader.dataset.block_size
    throughput = total_tokens / (end_time - start_time)

    print(f"state: training    epoch: {epoch}    running loss in this epoch: {running_loss_in_this_epoch}    accuracy in this epoch: {accuracy_in_this_epoch:.2f}    grad_norm: {avg_grad_norm:.4f}    throughput: {throughput:.2f} tokens/s")

    return running_loss_in_this_epoch, accuracy_in_this_epoch, avg_grad_norm, throughput

def validate(model, dataloader, criterion, epoch, device):   
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device).to(torch.long)
            output = model(decoder_input=data)
            loss = criterion(output.view(-1, output.shape[2]), label.view(-1))
            total_loss += loss.item()
            correct += torch.argmax(output, dim=2).eq(label).sum().item()
            total += data.shape[0] * data.shape[1]
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"state: validate    epoch: {epoch}    loss: {avg_loss:.4f}    accuracy: {accuracy:.2f}")
    return avg_loss, accuracy

def main():
    script_dir = os.path.dirname(__file__)

    config = load_config(os.path.join(script_dir, "config.json"))

    meta_path = os.path.join(script_dir, config["data_path"]["meta"])
    train_path = os.path.join(script_dir, config["data_path"]["train"])
    val_path = os.path.join(script_dir, config["data_path"]["val"]) 

    assert os.path.exists(meta_path), f"{meta_path} doesn't exist!"
    assert os.path.exists(train_path), f"{train_path} doesn't exist!"
    assert os.path.exists(val_path), f"{val_path} doesn't exist!"

    with open(meta_path, "rb") as f:
        data = pickle.load(f)
        num_of_words_in_decoder = data["vocab_size"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(train_path, "rb") as f:
        train_text = np.fromfile(f, dtype=np.int32)
    with open(val_path, "rb") as f:
        val_text = np.fromfile(f, dtype=np.int32)

    train_dataset = mydataset(torch.from_numpy(train_text).to(torch.int), config["block_size"])
    val_dataset = mydataset(torch.from_numpy(val_text).to(torch.int), config["block_size"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=torch.cuda.is_available())

    assert config["max_len"] > config["block_size"], "max_len must be greater than block_size"

    save_model_path = os.path.join(script_dir, config["save_model_path"])
    model = transformers_model(config["mode"], config["is_same_language"], config["embedding_dimension"], config["num_of_attention_head"], config["num_of_words_in_encoder"], num_of_words_in_decoder, config["hidden_units_dimension"], config["encoder_layers"], config["decoder_layers"], config["dropout"], config["max_len"])
    if os.path.exists(save_model_path):
        with open(save_model_path, "rb") as f:
            model.load_state_dict(torch.load(f))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = get_transformer_scheduler(optimizer)

    writer = SummaryWriter('runs/experiment')

    for epoch in range(1, config["num_epochs"] + 1):
        train_loss, train_acc, avg_grad_norm, throughput = train_one_epoch(
            model, train_loader, criterion, scheduler, epoch, device, save_model_path
        )
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)

        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Gradient_Norm', avg_grad_norm, epoch)
        writer.add_scalar('Throughput', throughput, epoch)

    writer.close()

if __name__ == "__main__":
    main()