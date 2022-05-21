import argparse
import dgl
import json
import math
import os
import pickle
import torch
import random
import numpy as np

from model_cls import classifier
from sklearn.metrics import  accuracy_score,f1_score
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

num_gpus = 1
local_rank = 0

DEVICE = torch.device("cuda", local_rank)

dataset = './utils/data.pkl'
with open(dataset,'rb') as f:
    data = pickle.load(f)
word_dict = data['word_dict']
print(list(word_dict.keys())[0:10],type(list(word_dict.keys())[0]))
soundfeature_encoder = data['phonetic_features']
soundposition_encoder = data['phonetic_pronunciation']
oldsoundgraphs = data['phonetic_graphs']
bert2char_encoder = data['bert2character']
soundgraphs = {}
for k in range(len(oldsoundgraphs)):
    soundgraphs[k] = dgl.from_scipy(oldsoundgraphs[k]).to(DEVICE)
meanfeature_encoder = data['semantic_features']
meanposition_encoder = data['semantic_positions']
oldmeangraphs = data['semantic_graphs']
meangraphs = {}
for k in range(len(oldmeangraphs)):
    meangraphs[k] = dgl.from_scipy(oldmeangraphs[k]).to(DEVICE)
temp_prongraph1 = data['proconnection_graph'] #414
PRONUNCIATIONGRAPH = dgl.from_scipy(temp_prongraph1).to(DEVICE)


class pyTokenizer(object):
    def __init__(self, config):
        super(pyTokenizer, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(config['chinese_bert_path'], do_lower_case=True)
        self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.max_length = config['max_length']
    
    def encode(self, text):
        text1 = self.tokenizer.encode(text, max_length=self.max_length, pad_to_max_length=True, truncation=True)
        return text1



def getItems(Dict, keys):
    return [Dict[k] for k in keys]


class TabularDataset(Dataset):
    def __init__(self, file, flag, format, config, skip_header=False):
        super(TabularDataset).__init__()
        self.tokenizer= pyTokenizer(config)
        f = None
        self.label_dict = None
        self.num_label = 0
        label_dict = {}
        if format == 'tsv':
            f = open(file,'r').read()
            f = f.split('\n')
            if len(f[-1]) == 0:
                f = f[:-1]
            if skip_header == True:
                f = f[1:]
        if flag == 'train':
            if num_gpus != 1:
                per_worker = int(math.ceil(len(f) / float(num_gpus)))
                worker_id = local_rank
                iter_start = worker_id * per_worker
                iter_end = min(iter_start + per_worker, len(f))
                f = f[iter_start:iter_end]
        if format == 'tsv':
            lines = [iii.split('\t')+[1] for iii in f]
            label_list = ['0','1']
            self.num_label = len(label_list)
            self.label_dict = {ind:{"label": label} for ind, label in enumerate(label_list)}
        self.newTexts  = []
        
        for i in lines:
            inOrOut = []                   # max_length
            soundIndex = []                # not sure
            charPositionInSoundIndex = []  # numChinese
            soundPositionIndex = []        # same as soundIndex
            soundGraph = []                # numChinese
            meanIndex = []                 # not sure
            charPositionInMeanIndex = []   # numChinese
            meanPositionIndex = []         # same as meanIndex
            meanGraph = []                 # numChinese
            
            pronunciationIndex = []                 # numChinese
            label = int(i[0])
            index = int(i[2])
            text = self.tokenizer.encode(i[1])
            chineseCount = 0
            for char in text:
                if word_dict[char] == 0:
                    inOrOut.append(-1)
                else:
                    inOrOut.append(chineseCount)
                    chineseCount += 1
                    charIndex = bert2char_encoder[char]
                    tempSound = soundfeature_encoder[charIndex][1:soundfeature_encoder[charIndex][0]].copy()
                    tempSound[1:] += 6763
                    charPositionInSoundIndex.append(len(soundIndex))
                    soundIndex.extend(tempSound)
                    soundPositionIndex.extend(soundposition_encoder[charIndex][1:soundposition_encoder[charIndex][0]].copy())
                    soundGraph.append(charIndex)

                    tempmean = meanfeature_encoder[charIndex][1:meanfeature_encoder[charIndex][0]].copy()
                    tempmean[1:] += 8206
                    charPositionInMeanIndex.append(len(meanIndex))
                    meanIndex.extend(tempmean)
                    meanPositionIndex.extend(meanposition_encoder[charIndex][1:meanposition_encoder[charIndex][0]].copy())
                    meanGraph.append(charIndex)

                    pronunciationIndex.append(charIndex)
            self.newTexts.append([label, text, inOrOut, soundIndex,soundPositionIndex,meanIndex,meanPositionIndex,\
                soundGraph,meanGraph,charPositionInSoundIndex,charPositionInMeanIndex,\
                pronunciationIndex,index])
    def __len__(self):
        return len(self.newTexts)

    def __getitem__(self, index):
        return self.newTexts[index]

    def get_label_dict(self):
        return self.label_dict
    
    def get_num_label(self):
        return self.num_label


class graphDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super(graphDataLoader).__init__()
        self.batch_size = batch_size
        self.dataset = []
        self.label_dict = dataset.get_label_dict()
        self.num_label = dataset.get_num_label()
        self.length = math.ceil(len(dataset)/batch_size)
        for i in range(self.length):
            if i != (self.length-1):
                temp = dataset[i*batch_size: (i+1)*batch_size]
            else:
                temp = dataset[i*batch_size:]
            labels = []
            text = []
            inOrOut = []
            soundIndex = []
            soundPositionIndex = []
            meanIndex = []
            meanPositionIndex = []
            soundGraph = []
            meanGraph = []
            charPositionInSoundIndex = []
            charPositionInMeanIndex = []
            pronunciationIndex = []
            indexs = []
            for j in range(len(temp)):
                labels.append(temp[j][0])
                text.append(np.array(temp[j][1]))
                inOrOut.extend([ k if (k == -1) else (k+len(soundGraph)) for k in temp[j][2]])
                charPositionInSoundIndex.extend([ (k+len(soundIndex)) for k in temp[j][9]])
                charPositionInMeanIndex.extend([ (k+len(soundIndex)) for k in temp[j][10]])
                pronunciationIndex.extend(temp[j][11])
                soundIndex.extend(temp[j][3])
                soundPositionIndex.extend(temp[j][4])
                meanIndex.extend(temp[j][5])
                meanPositionIndex.extend(temp[j][6])
                soundGraph.extend(temp[j][7])
                meanGraph.extend(temp[j][8])
                indexs.append(temp[j][12])
            labels = torch.tensor(labels).to(DEVICE)
            text = torch.tensor(text).to(DEVICE)
            inOrOut = torch.tensor(inOrOut).to(DEVICE)
            charPositionInSoundIndex = torch.tensor(charPositionInSoundIndex).to(DEVICE)
            charPositionInMeanIndex = torch.tensor(charPositionInMeanIndex).to(DEVICE)
            pronunciationIndex = torch.tensor(pronunciationIndex).to(DEVICE)
            soundIndex = torch.tensor(soundIndex).to(DEVICE)
            soundPositionIndex = torch.tensor(soundPositionIndex).to(DEVICE)
            meanIndex = torch.tensor(meanIndex).to(DEVICE)
            meanPositionIndex = torch.tensor(meanPositionIndex).to(DEVICE)
            pronunciationGraph = None
            indexs = torch.tensor(indexs).to(DEVICE)
            self.dataset.append([labels, [text, inOrOut, soundIndex,soundPositionIndex,meanIndex,meanPositionIndex, \
                soundGraph,meanGraph,pronunciationGraph,charPositionInSoundIndex,charPositionInMeanIndex,pronunciationIndex,indexs]])

    def __iter__(self):
        return iter(self.dataset)
    
    def get_label_name(self,i):
        return self.label_dict[i]

    def get_num_label(self):
        return self.num_label



def load_data(config):
    train = None
    dev = None
    test = None
    postfix = 'tsv'
    train = TabularDataset(config['data_dir']+"{}/train.{}".format(config['task'],postfix), 'train', postfix, config, True)
    dev = TabularDataset(config['data_dir']+"{}/dev.{}".format(config['task'],postfix), 'dev', postfix, config, True)
    test = TabularDataset(config['data_dir']+"{}/test.{}".format(config['task'],postfix), 'test', postfix, config, True)
    train_iter = graphDataLoader(train, config['train_batch_size'])
    dev_iter = graphDataLoader(dev, config['dev_batch_size'])
    test_iter = graphDataLoader(test, config['test_batch_size'])
    return train_iter, dev_iter, test_iter

def load_data_test(config):
    test = None
    postfix = 'tsv'
    test = TabularDataset(config['data_dir']+"{}/test.{}".format(config['task'],postfix), 'test',postfix, config, True)
    test_iter = graphDataLoader(test, config['test_batch_size'])
    return test_iter

def load_model(config, n_gpu=0, train_loader_length=0):
    model = classifier(config, num_labels = 2)
    model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    if local_rank == 0:
        for name, param in model.named_parameters():
            print(name,param.data.shape)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,lr=config['learning_rate'])
    if train_loader_length != 0:
        total_steps =  train_loader_length*config['num_train_epochs']
        num_warmup_steps = total_steps*0.1
        warmup_scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps,total_steps)
        return model, optimizer, warmup_scheduler
    else:
        return model, optimizer
def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, warmup_scheduler):
    EARLYSTOP = 0
    FLAG = 0
    training_losses = []
    validate_losses = []
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000

    test_acc = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0
    test_loss = 1000000000000000
    for idx in range(int(config['num_train_epochs'])):
        if FLAG == 1:
            break
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 10)
        print("EPOCH: ", str(idx))
        for _, batch in tqdm(enumerate(train_dataloader)):
            (label_ids, input_ids) = batch
            if config['tokenized']:
                input_ids = input_ids[0]
            new_input_ids = input_ids.copy()
            new_input_ids[6] = dgl.batch(getItems(soundgraphs, new_input_ids[6]))
            new_input_ids[7] = dgl.batch(getItems(meangraphs, new_input_ids[7]))
            new_input_ids[8] = PRONUNCIATIONGRAPH
            loss, _= model(new_input_ids, label_ids)
            loss.backward()
            tr_loss += loss.item()

            nb_tr_examples += label_ids.size(0)
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()
            warmup_scheduler.step()
            global_step += 1

            if (nb_tr_steps % config['checkpoint'] == 0) and (local_rank == 0):
                print("-*-" * 15)
                print("training loss: ")
                print(loss.item())
                training_losses.append(loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config['tokenized'], 'dev', config)
                print("......" * 10)
                print("dev: loss, acc, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_f1)
                validate_losses.append(tmp_dev_loss)
                EARLYSTOP += 1

                if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                    EARLYSTOP = 0
                    dev_best_acc = tmp_dev_acc
                    dev_best_loss = tmp_dev_loss
                    dev_best_f1 = tmp_dev_f1
                    eval_checkpoint(model, test_dataloader, config['tokenized'],'test', config)

                    if config['export_model']:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(config['output_dir'], "{}/pytorch_model.bin".format(config['task']))
                        
                        torch.save(model_to_save.state_dict(), output_model_file)
                print("-*-" * 15)
                model.train()
                if EARLYSTOP > config['earlystop']:
                    FLAG = 1
                    break

                
    
    if (local_rank == 0):
        print("-*-" * 15)
        print("training loss: ")
        print(loss.item())
        training_losses.append(loss.item())
        tmp_dev_loss, tmp_dev_acc, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config['tokenized'],'dev', config)
        print("......" * 10)
        print("dev: loss, acc, f1")
        print(tmp_dev_loss, tmp_dev_acc, tmp_dev_f1)
        validate_losses.append(tmp_dev_loss)
        if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
            dev_best_acc = tmp_dev_acc
            dev_best_loss = tmp_dev_loss
            dev_best_f1 = tmp_dev_f1
            eval_checkpoint(model, test_dataloader, config['tokenized'],'test', config)

            if config['export_model']:
                model_to_save = model.module if hasattr(model, "module") else model
                output_model_file = os.path.join(config['output_dir'], "{}/pytorch_model.bin".format(config['task']))
                torch.save(model_to_save.state_dict(), output_model_file)

        print("=&=" * 15)
        print("dev: current best precision, recall, f1, acc, loss ")
        print(dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc, dev_best_loss)
        print("test: current precision, recall, f1, acc, loss ")
        print(test_precision, test_recall, test_f1, test_acc, test_loss)
        print("=&=" * 15)
        with open('results/{}/main.out'.format(config['task']),'a') as outfi:
            outfi.write("ml:{}, bs:{}, lr:{}, epoch:{}, checkpoint:{}, wd:{}, head:{}, dropout:{}, seed:{} \n".format(\
                config['max_length'],config['train_batch_size'],config['learning_rate'],config['num_train_epochs'],\
                config['checkpoint'],config['weight_decay'],config['num_attention_heads'],config['dropout'],config['random_seed']))
            outfi.write("dev: f1={}, acc={}, loss={}||test: f1={}, acc={}, loss={}.\n\n".format(\
                dev_best_f1, dev_best_acc, dev_best_loss,test_f1, test_acc, test_loss))
    return



def eval_checkpoint(model_object, eval_dataloader, tokenized, flag, config):
    model_object.eval()

    eval_loss = 0
    eval_accuracy = 0
    eval_f1 = []
    logits_all = []
    labels_all = []
    eval_steps = 0
    for batch in tqdm(eval_dataloader):
        (label_ids, input_ids) = batch
        if tokenized:
            input_ids = input_ids[0]
        new_input_ids = input_ids.copy()
        new_input_ids[6] = dgl.batch(getItems(soundgraphs, new_input_ids[6]))
        new_input_ids[7] = dgl.batch(getItems(meangraphs, new_input_ids[7]))
        new_input_ids[8] = PRONUNCIATIONGRAPH
        with torch.no_grad():
            tmp_eval_loss, logits = model_object(new_input_ids, label_ids)

        
        label_ids = label_ids.to("cpu").numpy()
        logits = logits.detach().cpu().numpy()
        eval_loss += tmp_eval_loss.mean().item()

        logits_all.extend(logits.tolist())
        labels_all.extend(label_ids.tolist())
        eval_steps += 1

    average_loss = round(eval_loss / eval_steps, 4)
    eval_accuracy = float(accuracy_score(y_true=labels_all, y_pred=logits_all))
    eval_f1 = float(f1_score(y_true=labels_all, y_pred=logits_all, average='micro'))
    return average_loss, eval_accuracy, eval_f1 



def main():
    parser = argparse.ArgumentParser(description='Get task name.')
    parser.add_argument('-seed', default=42, type=int, help='random seed')
    parser.add_argument('-task', default="resume", type=str)
    parser.add_argument('--local_rank',type=int)
    parser.add_argument('-weight_decay', default=0.01, type=float)
    parser.add_argument("-learning_rate", default=5e-5, type=float)
    parser.add_argument("-dropout", default=0.5, type=float)
    args = parser.parse_args()
    task = args.task
    config = json.load(open('./config/config_cls.json'))
    config['task'] = task
    config['random_seed'] = args.seed
    config['weight_decay'] = args.weight_decay
    config['learning_rate'] = args.learning_rate
    config['dropout'] = args.dropout
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    dgl.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    if local_rank == 0:
        for key in config.keys():
            print("{}: {}".format(key,config[key]))
    if config['flag'] == 'train':
        train_loader, dev_loader, test_loader = load_data(config)
        model, optimizer, warmup_scheduler = load_model(config, num_gpus,train_loader.length)
        train(model, optimizer, train_loader, dev_loader, test_loader, config, warmup_scheduler)
    elif config['flag'] == 'test':
        test_loader = load_data_test(config)
        model, optimizer = load_model(config, num_gpus)
        model.load_state_dict(torch.load(os.path.join(config['output_dir'], "{}/pytorch_model.bin".format(config['task']))))
        eval_checkpoint(model, test_loader, config['tokenized'], config)
    return

if __name__ == "__main__":
    main()