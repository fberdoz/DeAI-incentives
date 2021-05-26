import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from random import sample
from itertools import chain, combinations
from math import factorial

#########################################################
#                  Federated Learning                   #
#########################################################

def client_update(client_model, optimizer, criterion, train_loader, epoch=5):
    """Train a client_model on the train_loder data. Found in the DeAI repository."""
    client_model.train()
    for e in range(epoch):
        for batch_idx, (target, cont_data, cat_data) in enumerate(train_loader):
            optimizer.zero_grad()
            output = client_model(cont_data, cat_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def diffuse_params(client_models, communication_matrix):
    """Diffuse the models with their neighbors. Found in the DeAI repository."""
    if client_models:
        client_state_dicts = [model.state_dict() for model in client_models]
        keys = client_state_dicts[0].keys()
    for model, weights in zip(client_models, communication_matrix):
        neighbors = np.nonzero(weights)[0]
        model.load_state_dict(
            {
                key: torch.stack(
                    [weights[j]*client_state_dicts[j][key] for j in neighbors],
                    dim=0,
                ).sum(0) / weights.sum() 
                for key in keys
            }
        )

def average_models(global_model, client_models):
    """Average models across all clients. Found in the DeAI repository."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

#########################################################
#               Performance Monitoring                  #
#########################################################

def evaluate_model(model, data_loader, criterion=None, threshold=0.5):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    n = len(data_loader.dataset)
    if n == 0:
        raise ValueError('Empty dataset')
    loss = 0
    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    pos = 0
    pred_pos = 0
    
    with torch.no_grad():
        for target, cont_data, cat_data in data_loader:
            output = model(cont_data, cat_data)
            
            if criterion is not None:
                loss += criterion(output, target).item()  # sum up batch loss
            
            pred = torch.where(output > threshold, torch.tensor([1]), torch.tensor([0]))
            
            correct += pred.eq(target).sum().item()
            TP += pred[pred==1].eq(target[pred==1]).sum().item()
            TN += pred[pred==0].eq(target[pred==0]).sum().item()
            FP += pred[pred==1].ne(target[pred==1]).sum().item()
            FN += pred[pred==0].ne(target[pred==0]).sum().item()
            
            pos += target.sum().item()
            pred_pos += pred.sum().item()
    
    neg = n - pos
    pred_neg = n - pred_pos
    
    # Loss (only computed if criterion is given)
    if criterion is not None:
        loss_norm = loss / n
    else:
        loss_norm = None
    acc = correct / n
    
    # precision
    if pred_pos != 0:
        precision = TP / pred_pos
    else:
        precision = 1
    
    # recall or true positive rate (TPR)
    if pos != 0:
        recall = TP / pos
    else:
        recall = 1
            
    # f1 score
    if precision != 0 and recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else: 
        f1 = 0
    
    # fall-out or false positive rate (FPR)
    if neg == 0:
        FPR = 1
    else:
        FPR = FP / neg
        
        
    perf = {
        'loss_norm': loss_norm,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'FPR': FPR
    }
    return perf

def evaluate_many_models(models, data_loader, criterion, threshold):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_nodes = len(models)
    performances = []
    for i in range(num_nodes):
        perfs += [evaluate_model(models[i], data_loader, criterion, threshold)]
    return performances

def initialize_perf(sizes):
    'Initialize the performance dictionary, for code clarity.'
    perf = {
        'loss_norm': np.zeros(sizes),
        'accuracy': np.zeros(sizes),
        'precision': np.zeros(sizes),
        'recall': np.zeros(sizes),
        'f1' : np.zeros(sizes),
        'FPR': np.zeros(sizes)
    }
    return perf

def fill_perf_history(perf, perf_hist, index):
    'Fill an history dictionary with another dictionary with the same keys.'
    for key in perf:
        perf_hist[key][index] = perf[key]
        
#########################################################
#                     Data Loading                      #
#########################################################
# The following functions are strongly inspired by https://www.kaggle.com/chriszou/titanic-with-pytorch-nn-solution


class TabularDataset(torch.utils.data.Dataset):
    """Create a torch dataset using a dataframe as an input."""
    def __init__(self, df, categorical_columns, output_column=None):
        super().__init__()
        self.len = df.shape[0]
        
        # Store categorical and continuous data separately
        self.categorical_columns = categorical_columns
        self.continous_columns = [col for col in df.columns if col not in self.categorical_columns + [output_column]]
        
        if self.continous_columns:
            self.cont_X = df[self.continous_columns].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.len, 1))

        if self.categorical_columns:
            self.cat_X = df[self.categorical_columns].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.len, 1))

        if output_column != None:
            self.has_label = True
            self.label = df[output_column].astype(np.float32).values.reshape(-1, 1)
        else:
            self.has_label = False

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.has_label:
              return [self.label[index], self.cont_X[index], self.cat_X[index]]
        else:
              return [self.cont_X[index], self.cat_X[index]]
            
            
def load_into_df(dataset, drop=[], na="mean_feature"):
    """Load the given dataset into panda dataframes, and returns some useful metadata."""
    
    if dataset == 'titanic':
        # Loading the data
        train_data = pd.read_csv('./data/titanic/train.csv')
        test_data_unlab = pd.read_csv('./data/titanic/test.csv') # unlabelled, not used

        # Concatenation for same preprocessing
        all_df = pd.concat([train_data, test_data_unlab], sort=False)

        # Categorical features
        cat_cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
        cont_cols = ['Age', 'Fare']

        # Data preprocessing
        all_df = all_df.drop(drop, axis=1)

        for cat_col in cat_cols:
            if cat_col in ['Embarked']:
                all_df[cat_col] = LabelEncoder().fit_transform(all_df[cat_col].astype(str))
            else:
                all_df[cat_col] = LabelEncoder().fit_transform(all_df[cat_col])

        # Filling the empty cells using the specified mode.
        if na == 'mean_feature':
            all_df = all_df.fillna(all_df.mean()) 

        elif na == 'discard_sample':
            raise NotImplementedError

        elif na == 'discard_feature':
            raise NotImplementedError

        else:
            raise ValueError('Unkown argument: {}.'.format(na))

        # Metadata for centralizing information
        cat_dims = [int(all_df[col].nunique()) for col in cat_cols] # Numbers of categories for each caterorical features
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims] # Dimension of embeddings for each category
        n_cont = 2 # number of continuous features
        n_feat = n_cont + sum([y for x, y in emb_dims])
        cont_cols = list(set(all_df.columns) - set(cat_cols))
        
        meta = {
            "cat_cols" : cat_cols,
            "cont_cols": cont_cols,
            "cat_dims" : cat_dims,
            "emb_dims" : emb_dims,
            "n_cont" : n_cont,
            "n_feat" : n_feat,
            "label" : 'Survived'
        }

        # Deconcatenating
        train_df = all_df.head(train_data.shape[0])
        test_df_unlab = all_df.tail(test_data_unlab.shape[0])

        return train_df, test_df_unlab, meta
    
    elif dataset == 'income':
        # Loading the data
        all_df = pd.read_csv('./data/income/adult.csv')

        # Categorical/continuous features
        cat_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race',
                    'gender', 'native-country']
        cont_cols = ['educational-num', 'fnlwgt', 'capital-loss', 
                     'capital-gain', 'hours-per-week', 'age']
        
        label = 'income'
        
        # Data preprocessing
        all_df = all_df.drop(drop, axis=1)
        #all_df.replace('?', np.nan, inplace=True)
        all_df.replace({'income': '<=50K'}, 0., inplace=True)
        all_df.replace({'income': '>50K'}, 1., inplace=True)
        
        for cat_col in cat_cols:
            all_df[cat_col] = LabelEncoder().fit_transform(all_df[cat_col])

        # Filling the empty cells using the specified mode.
        if na == 'mean_feature':
            all_df = all_df.fillna(all_df.mean()) 

        elif na == 'discard_sample':
            raise NotImplementedError

        elif na == 'discard_feature':
            raise NotImplementedError

        else:
            raise ValueError('Unkown argument: {}.'.format(na))

        # Metadata for centralizing information
        cat_dims = [int(all_df[col].nunique()) for col in cat_cols] # Numbers of categories for each caterorical features
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims] # Dimension of embeddings for each category
        n_cont = 6 # number of continuous features
        n_feat = n_cont + sum([y for x, y in emb_dims])
        
        meta = {
            "cat_cols" : cat_cols,
            "cont_cols": cont_cols,
            "cat_dims" : cat_dims,
            "emb_dims" : emb_dims,
            "n_cont" : n_cont,
            "n_feat" : n_feat,
            "label" : label
        }

        return all_df, meta
    else:
        raise ValueError('Unkown dataset')

def df_to_datasets(data_df, num_clients, meta, f_test=0.2, f_tot=1):
    """Split the full dataframe into several datasets (one per clients)."""
    
    # Full dataset
    full_ds = TabularDataset(data_df, meta['cat_cols'], meta['label'])
    N_full = len(full_ds)
    
    # Reduced dataset (for code verification)
    reduced_ds = torch.utils.data.random_split(full_ds, [int(f_tot * len(data_df)), len(data_df) - int(f_tot * len(data_df))])[0]
    N_red = len(reduced_ds)
    
    # Creating the size list for each dataset
    if isinstance(num_clients, int) and num_clients > 0:
        sizes = [int(1/num_clients * N_red) for _ in range(num_clients)]
    
    elif isinstance(num_clients, list) and len(num_clients) > 0 and sum(num_clients) <= 1.0 and all(sz > 0 for sz in num_clients):
        sizes = [int(f * N_red) for f in num_clients]

    else:
        raise ValueError("""Argument 'num_clients' is of wrong type or value.\nMust be a positive
                            integer or a nonempty list of positive float summing to at most 1.\n
                            Currently of type {}.""".format(type(num_clients)))
               
    # Splitting the datatests
    if (N_red - sum(sizes)) != 0:
        split_ds = torch.utils.data.random_split(reduced_ds, sizes + [N_red - sum(sizes)])
        res = split_ds.pop()
    else:
        split_ds = torch.utils.data.random_split(reduced_ds, sizes)
    
    # Splitting each datasets into a training and testing dataset

    train_ds = []
    test_ds = []
    
    N_tr = 0
    N_te = 0
    
    for ds in split_ds:
        N_te_i = int(f_test * len(ds))
        N_tr_i = len(ds) - N_te_i
        
        train_ds_i, test_ds_i = torch.utils.data.random_split(ds, [N_tr_i, N_te_i])
        train_ds.append(train_ds_i)
        test_ds.append(test_ds_i)
        
        N_tr += N_tr_i
        N_te += N_te_i
        
    print("Sizes:\nFull dataset: {}\nReduced dataset: {}\nSplitted datasets (total): {} (train: {}, test: {})\nPer client: {}"
          .format(N_full, N_red, sum(sizes), N_tr, N_te, sizes))
    return  train_ds, test_ds

def df_to_ds(data_df, sizes, meta, f_test=0.2, x_noise=None, x_acc=None, y_acc=None):
    """Split the full dataframe into several datasets (one per clients)."""
    
    # Full dataset
    full_ds = TabularDataset(data_df, meta['cat_cols'], meta['label'])
    N_full = len(full_ds)
    
    # Creating the size list for each dataset
    if isinstance(sizes, int) and sizes > 0:
        sizes = [int(1/sizes * N_full) for _ in range(sizes)]

    elif isinstance(sizes, list) and len(sizes) > 0 and sum(sizes) <= 1.0 and all(sz > 0 for sz in sizes):
        sizes = [int(f * N_full) for f in sizes]

    else:
        raise ValueError("""Argument 'num_clients' is of wrong type or value.\nMust be a positive
                            integer or a nonempty list of positive float summing to at most 1.\n
                            Currently of type {}.""".format(type(num_clients)))
               
    # Splitting the datatests
    if (N_full - sum(sizes)) != 0:
        split_ds = torch.utils.data.random_split(full_ds, sizes + [N_full - sum(sizes)])
        res = split_ds.pop()
    else:
        split_ds = torch.utils.data.random_split(full_ds, sizes)
    
    # Altering features
    # Continuous features
    if x_noise is not None:
        assert isinstance(x_noise, list) and len(x_noise) == len(sizes)
        raise NotImplemetedError
        
    # Categorical features   
    if x_acc is not None:
        assert isinstance(x_acc, list) and len(x_acc) == len(sizes)
        raise NotImplemetedError
    
    # Altering labels
    if y_acc is not None:
        assert isinstance(y_acc, list) and len(y_acc) == len(sizes)
        for i, ds in enumerate(split_ds):
            n2change = int((1.0 - y_acc[i]) * len(ds))
            idx2change = sample(ds.indices, n2change)
            full_ds.label[idx2change] = 1.0 - full_ds.label[idx2change]
        
    # Splitting each datasets into a training and testing dataset
    train_ds = []
    test_ds = []
    
    N_tr = 0
    N_te = 0
    
    for ds in split_ds:
        N_te_i = int(f_test * len(ds))
        N_tr_i = len(ds) - N_te_i
        
        train_ds_i, test_ds_i = torch.utils.data.random_split(ds, [N_tr_i, N_te_i])
        train_ds.append(train_ds_i)
        test_ds.append(test_ds_i)
        
        N_tr += N_tr_i
        N_te += N_te_i
        
    print("Sizes:\nFull dataset: {}\nSplitted datasets (total): {} (train: {}, test: {})\nPer client: {}"
          .format(N_full, sum(sizes), N_tr, N_te, sizes))
    return  train_ds, test_ds


def ds_to_dataloaders(dataset, batch_size):
    """Create a (list of) torch dataloader(s) given a (list of) dataset(s)"""
    if isinstance(dataset, list):
        dl = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in dataset]
    elif isinstance(dataset, torch.utils.data.dataset.Subset) or isinstance(dataset, torch.utils.data.dataset.ConcatDataset):
        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        raise TypeError("Argument 'dataset' is of type {}. Must be a torch dataset or a list of torch datasets.".format(type(dataset)))
    
    return dl

def coalition_dataloader(coal, list_ds, batch_size):
    return ds_to_dataloaders(torch.utils.data.ConcatDataset([test_ds[i] for i in coal]), batch_size)

def normalize(df, meta):
    df_norm = df.copy()
    cont_cols = list(set(df.columns) - set(meta['cat_cols']))
    df_norm[meta['cont_cols']] = (df[meta['cont_cols']] - df[meta['cont_cols']].mean()) / df[meta['cont_cols']].std()
    
    return df_norm

#########################################################
#                    Contributions                      #
#########################################################

def marginal_contribution(perf, marg_perfs, profit=False):
    if profit:
        contributions = perf - marg_perfs
    else:
        contributions = marg_perfs - perf
    return contributions

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def ShapleyValues(n, coal_models, marg_loaders, metric, criterion, threshold, profit=False):

    sv = np.zeros(n)
    
    for client_id in range(n):
        for coal in powerset(range(n)):
            if client_id in coal:
                
                
                coal_minus_client = list(coal)
                coal_minus_client.remove(client_id)
                coal_minus_client = tuple(coal_minus_client)
                
                coeff = factorial(n-len(coal)) * factorial(len(coal)-1) / factorial(n)
                
                perf_with_client = evaluate_model(coal_models[coal], marg_loaders[client_id], criterion, threshold)[metric]
                perf_minus_client = evaluate_model(coal_models[coal_minus_client], marg_loaders[client_id], criterion, threshold)[metric]
                
                if profit:
                    sv[client_id] += coeff * (perf_with_client - perf_minus_client)
                else:
                    sv[client_id] += coeff * (perf_minus_client - perf_with_client)
                        
    return sv