# MIT License
# 
# Copyright (c) 2024 Zihan Zhang, Yi Zhao, Harbin Institute of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import math
import json
import torch
import numpy
import pickle
import random
import argparse
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import get_linear_schedule_with_warmup
from eegcnn import EEGcnn, PositionalEncoding
from data_imagine import get_dataset
from myutils import kmeans
from collections import defaultdict, Counter
from sklearn.metrics import f1_score

class EEGclassification(torch.nn.Module):
    def __init__(self, chans=122, timestamp=165, cls=3, dropout1=0.1, dropout2=0.1, layer=0, pooling=None, size1=8, size2=8, feel1=125, feel2=25):
        super().__init__()
        self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2, F1=size1, D=size2, F2=size1*size2, P1=2, P2=5, dropoutRate=dropout1)
        self.linear = torch.nn.Linear(timestamp*size1*size2 if pooling is None else size1*size2, cls)
        self.layer = layer
        self.pooling = pooling
        if self.layer > 0:
            self.poscode = PositionalEncoding(size1*size2, dropout=dropout2, max_len=timestamp)
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=size1*size2, nhead=size1*size2//8, dim_feedforward=4*size1*size2, batch_first=True, dropout=dropout2), num_layers=self.layer)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        # assert inputs.shape == (inputs.shape[0], 122, 1651)
        hidden = self.eegcnn(inputs).permute(0, 2, 1)
        if self.layer > 0:        
            hidden = self.poscode(hidden)
            hidden = self.encoder(hidden, src_key_padding_mask=(mask.bool()==False))
        if self.pooling is None: hidden = torch.flatten(hidden, start_dim=1)
        if self.pooling == "mean": hidden = torch.sum(hidden*mask.unsqueeze(dim=2), dim=1)/torch.sum(mask, dim=1).unsqueeze(dim=1)
        if self.pooling == "sums": hidden = torch.sum(hidden*mask.unsqueeze(dim=2), dim=1)
        if self.pooling == "tops": hidden = hidden[:, 0, :]
        output = self.linear(hidden)
        # assert output.shape == (inputs.shape[0], 3)
        return output


class ImagineDecodeDataset(Dataset):
    def __init__(self, istrain, rand_guess, subject, textmaps):
        self.input_features = []
        self.labels = []
        data = get_dataset(subject)
        inputs = data["input_features"]
        labels = data["labels"]
        print("the length of inputs is {}".format(len(inputs)))
        for index in range(len(inputs)):
            if ((index % 5 == 1) ^ istrain) and textmaps[labels[index]] >= 0:
                self.input_features.append(inputs[index])
                self.labels.append(textmaps[labels[index]])
        print(len(self.input_features), len(self.labels), index, len(inputs), len(labels), Counter(self.labels))
        if rand_guess == 1: random.shuffle(self.input_features)

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        return self.input_features[idx], torch.tensor(self.labels[idx]), torch.ones(165)
    
    def sample_subset(self, subset_ratio):
        dataset_size = len(self.input_features)
        subset_size = int(dataset_size * subset_ratio)
        indices = list(range(dataset_size))
        random.shuffle(indices)
        sampled_indices = indices[:subset_size]
        self.input_features = [self.input_features[i] for i in sampled_indices]
        self.labels = [self.labels[i] for i in sampled_indices]


class ZucoDecodeDataset(Dataset):
    def __init__(self, istrain, rand_guess, textmaps, textlist):
        self.input_features = []
        self.labels = []
        self.length = []
        with open("zuco_dataset.pkl", "rb") as file: data = pickle.load(file)
        inputs = data[0]["input_features"]+data[1]["input_features"]+data[2]["input_features"]
        labels = data[0]["labels"]+data[1]["labels"]+data[2]["labels"]
        textdics = dict()
        for idx, i in enumerate(textlist): textdics[i] = idx
        for index in range(len(inputs)):
            if ((textdics[labels[index]] % 5 == 1) ^ istrain) and textmaps[labels[index]] >= 0:
                self.input_features.append(inputs[index][:, :5000] if inputs[index].shape[1] >= 5000 else torch.nn.functional.pad(inputs[index], (0, 5000-inputs[index].shape[1])))
                self.length.append(torch.ones(500) if inputs[index].shape[1] >= 5000 else torch.cat([torch.ones(inputs[index].shape[1]//10), torch.zeros(500-inputs[index].shape[1]//10)], dim=0))
                self.labels.append(textmaps[labels[index]])
                assert isinstance(self.input_features[-1], torch.Tensor) and self.input_features[-1].shape == (105, 5000)
                assert isinstance(self.length[-1], torch.Tensor) and self.length[-1].shape == (500,)
                assert isinstance(self.labels[-1], int)
            assert labels[index] in textmaps
            assert labels[index] in textdics
        print(len(self.input_features), len(self.labels), index, len(inputs), len(labels), Counter(self.labels))
        if rand_guess == 1: random.shuffle(self.input_features)

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        return self.input_features[idx], torch.tensor(self.labels[idx]), self.length[idx]


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr1', type=float, default=1e-3)
parser.add_argument('--wd1', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--train_log', type=int, default=10)
parser.add_argument('--evals_log', type=int, default=100)
parser.add_argument('--checkpoint_log', type=int, default=100)
parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
parser.add_argument('--chans', type=int, default=122)
parser.add_argument('--timestamp', type=int, default=165)
parser.add_argument('--pooling', type=str, default=None)
parser.add_argument('--size1', type=int, default=8)
parser.add_argument('--size2', type=int, default=8)
parser.add_argument('--feel1', type=int, default=125)
parser.add_argument('--feel2', type=int, default=25)
parser.add_argument('--cls', type=int, default=3)
parser.add_argument('--layer', type=int, default=0)
parser.add_argument('--dropout1', type=float, default=0.1)
parser.add_argument('--dropout2', type=float, default=0.1)
parser.add_argument('--sub', type=str, default='a')
parser.add_argument('--rand_guess', type=int, default=0) #Used to shuffle the correspondence between data input and labels to obtain random values
parser.add_argument('--dataset', type=str, default='imagine_decode')
parser.add_argument('--subset_ratio', type=float, default=1.0)
args = parser.parse_args()
print(args)

seed = args.seed
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = EEGclassification(chans=args.chans, timestamp=args.timestamp, cls=args.cls, dropout1=args.dropout1, dropout2=args.dropout2, layer=args.layer, pooling=args.pooling, size1=args.size1, size2=args.size2, feel1=args.feel1, feel2=args.feel2)


if args.dataset == "imagine_decode":
    with open("./Chisco/json/textmaps.json", "r") as file:
        textmaps_data = json.load(file)   
    textmaps = defaultdict(lambda: -1, textmaps_data)
    reversemaps = defaultdict(list)
    for i in textmaps: reversemaps[textmaps[i]].append(i)
    for i in reversemaps: print(i, len(reversemaps[i]), reversemaps[i])

    trainset = ImagineDecodeDataset(True, args.rand_guess, args.sub, textmaps)
    validset = ImagineDecodeDataset(False, False, args.sub, textmaps)
    trainset.sample_subset(args.subset_ratio)
    validset.sample_subset(args.subset_ratio)
    print(f"trainset[1] is {trainset[1]}")
    
if args.dataset == "zuco_decode":
    textlist = []
    textmaps = defaultdict(lambda: -1)
    with open("zuco1.txt") as file: textlist.extend([i.strip() for i in file.readlines()])
    with open("zuco2.txt") as file: textlist.extend([i.strip() for i in file.readlines()])
    with open("zuco4.txt") as file: textlist.extend([i.strip() for i in file.readlines()])
    with open("embeddingz.pkl", "rb") as file: embedding = pickle.load(file)
    print(len(textlist), numpy.array(embedding).shape)
    _, pred = kmeans(embedding, 20)
    pred = pred.tolist()
    for idx, i in enumerate(textlist): textmaps[i] = pred[idx]
    trainset = ZucoDecodeDataset(istrain=True, rand_guess=args.rand_guess, textmaps=textmaps, textlist=textlist)
    validset = ZucoDecodeDataset(istrain=False, rand_guess=False, textmaps=textmaps, textlist=textlist)


trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
validloader = DataLoader(validset, batch_size=args.batch, shuffle=True)
label_freqs = [0.0 for idx in range(args.cls)]
label_count = Counter(trainset.labels)
for i in label_count: label_freqs[i] = label_count[i]/len(trainset)
label_freqs = torch.tensor(label_freqs)
print(label_freqs)
print(len(trainset), len(trainloader))
print(len(validset), len(validloader))

def train(train_dataloader, valid_dataloader, model, config, label_frequency):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    label_frequency = torch.log(label_frequency.pow(config.tau)+1e-12).unsqueeze(dim=0)
    label_frequency = label_frequency.to(device)
    print(label_frequency.dtype, label_frequency.shape, label_frequency)

    training_step = len(train_dataloader)*config.epoch
    warmup_step = math.ceil(training_step*config.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr1, weight_decay=config.wd1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, training_step)

    running_loss = 0.0
    max_accuracy = 0.0  
    max_f1scores = 0.0
    for epoch in range(config.epoch):
        for idx, (input_features, labels, length) in enumerate(train_dataloader):
            step = epoch*len(train_dataloader)+idx+1
            model.train()

            input_features = input_features.to(device)
            labels = labels.to(device)
            length = length.to(device)
            # assert input_features.shape == (input_features.shape[0], 105, 5000)
            # assert labels.shape == (input_features.shape[0],)

            optimizer.zero_grad()
            output = model(input_features, length)
            loss = torch.nn.functional.cross_entropy(output+label_frequency, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if step % config.train_log == 0:
                print("step:{}(epoch{} {}/{}) loss:{}".format(step, epoch, idx, len(train_dataloader), running_loss/config.train_log))
                running_loss = 0.0
        
            if step % config.evals_log == 0:
                with torch.no_grad():
                    model.eval()
                    valid_output = []
                    valid_target = []
                    for idy, (valid_input_features, valid_labels, valid_length) in enumerate(valid_dataloader):
                        valid_input_features = valid_input_features.to(device)
                        valid_labels = valid_labels.to(device)
                        valid_length = valid_length.to(device)
                        # assert valid_input_features.shape == (valid_input_features.shape[0], 105, 5000)
                        # assert valid_labels.shape == (valid_input_features.shape[0],)
                        valid_output.append(model(valid_input_features, valid_length))
                        valid_target.append(valid_labels)
                    valid_output = torch.cat(valid_output, dim=0)
                    valid_target = torch.cat(valid_target, dim=0)
                    print(valid_output.shape, valid_target.shape)
                    valid_loss = torch.nn.functional.cross_entropy(valid_output+label_frequency, valid_target)
                    valid_accu = (torch.max(valid_output, dim=1)[1] == valid_target).float().mean()
                    valid_maf1 = f1_score(valid_target.tolist(), torch.max(valid_output, dim=1)[1].tolist(), average='macro')
                    max_accuracy = max(max_accuracy, valid_accu.item())
                    max_f1scores = max(max_f1scores, valid_maf1)
                print("step:{}(epoch{} {}/{}) valid_loss:{} accuracy:{} max_accuracy:{} f1:{} max_f1:{}".format(step, epoch, idx, len(train_dataloader), valid_loss.item(), valid_accu.item(), max_accuracy, valid_maf1, max_f1scores))

            if step % config.checkpoint_log == 0:
                print("saving model at step="+str(step)+"...")
                torch.save(model.state_dict(), config.checkpoint_path+"/checkpoint-"+str(step)+".pt")

    print("result:", max_accuracy, max_f1scores)

if not os.path.exists(args.checkpoint_path): os.mkdir(args.checkpoint_path)
train(trainloader, validloader, model, args, label_freqs)
