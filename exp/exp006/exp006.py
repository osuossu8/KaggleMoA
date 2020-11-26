### General ###
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
import sys
from scipy import stats

### Machine Learning ###
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

from pickle import load,dump
from tqdm import tqdm
tqdm.pandas()

sys.path.append("/root/workspace/KaggleMoA")
sys.path.append('../../iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from scipy.special import erfinv as sp_erfinv


def rank_gauss(data):
    epsilon = 1e-6

    for k in tqdm(GENES + CELLS):
        r_cpu = data.loc[:,k].argsort().argsort()
        r_cpu = (r_cpu/r_cpu.max()-0.5)*2 
        r_cpu = np.clip(r_cpu,-1+epsilon,1-epsilon)
        r_cpu = sp_erfinv(r_cpu) 
        data.loc[:,k] = r_cpu * np.sqrt(2)  
    return data


from torch.nn.modules.loss import _WeightedLoss


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
    
    
class LogitsLogLoss(Metric):

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 5e-5) + y_true * np.log(logits + 5e-5)
        return np.mean(-aux)


train_features = pd.read_csv('../../inputs/train_features.csv')
train_targets_scored = pd.read_csv('../../inputs/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../../inputs/train_targets_nonscored.csv')

test_features = pd.read_csv('../../inputs/test_features.csv')
df = pd.read_csv('../../inputs/sample_submission.csv')

train_features2=train_features.copy()
test_features2=test_features.copy()

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

# qt = QuantileTransformer(n_quantiles=100,random_state=42,output_distribution='normal')
# train_features[GENES+CELLS] = qt.fit_transform(train_features[GENES+CELLS])
# test_features[GENES+CELLS] = qt.transform(test_features[GENES+CELLS])
train_features = rank_gauss(train_features)
test_features = rank_gauss(test_features)


seed = 71

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)


n_comp = 600  #<--Update
pca_g = PCA(n_components=n_comp, random_state=seed)
data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
gpca = (pca_g.fit(data[GENES]))
train2 = (gpca.transform(train_features[GENES]))
test2 = (gpca.transform(test_features[GENES]))

train_gpca = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test_gpca = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train_gpca), axis=1)
test_features = pd.concat((test_features, test_gpca), axis=1)

dump(gpca, open('gpca.pkl', 'wb'))


#CELLS
n_comp = 50  #<--Update
pca_c = PCA(n_components=n_comp, random_state=seed)
data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
cpca= (pca_c.fit(data[CELLS]))
train2= (cpca.transform(train_features[CELLS]))
test2 = (cpca.transform(test_features[CELLS]))

train_cpca = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test_cpca = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

train_features = pd.concat((train_features, train_cpca), axis=1)
test_features = pd.concat((test_features, test_cpca), axis=1)

dump(cpca, open('cpca.pkl', 'wb'))
print('pca done')


from sklearn.decomposition import TruncatedSVD

n_comp_GENES = 450
n_comp_CELLS = 2

# GENES
data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
gsvd = (TruncatedSVD(n_components=n_comp_GENES, random_state=seed).fit(data[GENES]))
train2 = gsvd.transform(train_features[GENES]); test2 = gsvd.transform(test_features[GENES])

train_gsvd = pd.DataFrame(train2, columns=[f'svd_G-{i}' for i in range(n_comp_GENES)])
test_gsvd = pd.DataFrame(test2, columns=[f'svd_G-{i}' for i in range(n_comp_GENES)])

train_features = pd.concat((train_features, train_gsvd), axis=1)
test_features = pd.concat((test_features, test_gsvd), axis=1)

dump(gsvd, open('gsvd.pkl', 'wb'))

# CELLS
data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
csvd = ((TruncatedSVD(n_components=n_comp_CELLS, random_state=seed).fit(data[CELLS])))
train2 = csvd.transform(train_features[CELLS]); test2 = csvd.transform(test_features[CELLS])

train_csvd = pd.DataFrame(train2, columns=[f'svd_C-{i}' for i in range(n_comp_CELLS)])
test_csvd = pd.DataFrame(test2, columns=[f'svd_C-{i}' for i in range(n_comp_CELLS)])

train_features = pd.concat((train_features, train_csvd), axis=1)
test_features = pd.concat((test_features, test_csvd), axis=1)

dump(csvd, open('csvd.pkl', 'wb'))
print('svd done')


from sklearn.feature_selection import VarianceThreshold

c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
mask = (train_features[c_n].var() >= 0.85).values
tmp = train_features[c_n].loc[:, mask]
train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
tmp = test_features[c_n].loc[:, mask]
test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
print('Variance threshold done')


from sklearn.cluster import KMeans


def fe_cluster_genes(train, test, n_clusters_g = 22, SEED = seed):
    
    features_g = GENES
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_genes = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_genes, open('kmeans_genes.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    return train, test


def fe_cluster_cells(train, test, n_clusters_c = 4, SEED = seed):
    
    features_c = CELLS
    
    def create_cluster(train, test, features, kind = 'c', n_clusters = n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_cells = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_cells, open('kmeans_cells.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test


train_features2, test_features2=fe_cluster_genes(train_features2,test_features2)
train_features2, test_features2=fe_cluster_cells(train_features2,test_features2)
print('gene cell cluster done')


def fe_cluster_pca(train, test,n_clusters=5,SEED = seed):
        data=pd.concat([train,test],axis=0)
        kmeans_pca = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_pca, open('kmeans_pca.pkl', 'wb'))
        train[f'clusters_pca'] = kmeans_pca.predict(train.values)
        test[f'clusters_pca'] = kmeans_pca.predict(test.values)
        train = pd.get_dummies(train, columns = [f'clusters_pca'])
        test = pd.get_dummies(test, columns = [f'clusters_pca'])
        return train, test
    

train_pca=pd.concat((train_gpca,train_cpca),axis=1)
test_pca=pd.concat((test_gpca,test_cpca),axis=1)    
train_cluster_pca, test_cluster_pca = fe_cluster_pca(train_pca,test_pca)
train_cluster_pca = train_cluster_pca.iloc[:,650:]
test_cluster_pca = test_cluster_pca.iloc[:,650:]
print('cluster pca done')


def fe_cluster_svd(train, test,n_clusters=5,SEED = seed):
        data=pd.concat([train,test],axis=0)
        kmeans_pca = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_pca, open('kmeans_svd.pkl', 'wb'))
        train[f'clusters_svd'] = kmeans_pca.predict(train.values)
        test[f'clusters_svd'] = kmeans_pca.predict(test.values)
        train = pd.get_dummies(train, columns = [f'clusters_svd'])
        test = pd.get_dummies(test, columns = [f'clusters_svd'])
        return train, test


train_svd=pd.concat((train_gsvd,train_csvd),axis=1)
test_svd=pd.concat((test_gsvd,test_csvd),axis=1)
train_cluster_svd, test_cluster_svd = fe_cluster_svd(train_svd,test_svd)
train_cluster_svd = train_cluster_svd.iloc[:,470:]
test_cluster_svd = test_cluster_svd.iloc[:,470:]
print('cluster svd done')


train_features_cluster=train_features2.iloc[:,876:]
test_features_cluster=test_features2.iloc[:,876:]

gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167',
             'g-203','g-177','g-301','g-332','g-517','g-6', 'g-744','g-224','g-162','g-3','g-736','g-486',
             'g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298',
             'g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']


def fe_stats(train, test):
    
    features_g = GENES
    features_c = CELLS
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']
        
        
        for feature in features_c:
             df[f'{feature}_squared'] = df[feature] ** 2     
                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2        
        
    return train, test


train_features2,test_features2=fe_stats(train_features2,test_features2)
train_features_stats=train_features2.iloc[:,902:]
test_features_stats=test_features2.iloc[:,902:]
print('stats square done')


train_features = pd.concat((train_features, train_features_cluster,train_cluster_pca,train_features_stats), axis=1)
test_features = pd.concat((test_features, test_features_cluster,test_cluster_pca,test_features_stats), axis=1)
print('FE done')


train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]

train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)

target_cols = target.drop('sig_id', axis=1).columns.values.tolist()

target=target[target_cols]

train = pd.get_dummies(train, columns=['cp_time','cp_dose'])
test_ = pd.get_dummies(test, columns=['cp_time','cp_dose'])

feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['sig_id']]

train = train[feature_cols]
test = test_[feature_cols]

X_test = test.values


MAX_EPOCH = 200

tabnet_params = dict(
    n_d = 24, # 32,
    n_a = 24, # 32,
    n_steps = 1,
    gamma = 1.3,
    lambda_sparse = 0,
    optimizer_fn = optim.Adam,
    optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
    mask_type = "entmax",
    scheduler_params = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
    scheduler_fn = ReduceLROnPlateau,
    seed = seed,
    verbose = 10
)

scores_auc_all = []
test_cv_preds = []

NB_SPLITS = 10
mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = seed, shuffle = True)

oof_preds = np.zeros((len(train), len(target_cols)))
scores = []
scores_auc = []
SEED = [103, 104, 105, 106, 107, 108, 109]

for s in SEED:
    tabnet_params['seed'] = s
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train, target)):
        print("FOLDS: ", fold_nb + 1, 'seed:', tabnet_params['seed'])
        print('*' * 60)
    
        X_train, y_train = train.values[train_idx, :], target.values[train_idx, :]
        X_val, y_val = train.values[val_idx, :], target.values[val_idx, :]
        ### Model ###
        model = TabNetRegressor(**tabnet_params)
        
        ### Fit ###
        model.fit(
            X_train = X_train,
            y_train = y_train,
            eval_set = [(X_val, y_val)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs = MAX_EPOCH,
            patience = 20,
            batch_size = 1024, 
            virtual_batch_size = 32,
            num_workers = 1,
            drop_last = False,
            loss_fn = SmoothBCEwLogits(smoothing=5e-5))
        print('-' * 60)
    
        ### Predict on validation ###
        preds_val = model.predict(X_val)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        score = np.min(model.history["val_logits_ll"])
        saving_path_name = 'TabNet_seed_'+str(tabnet_params['seed'])+'_fold_'+str(fold_nb+1)
        saved_filepath = model.save_model(saving_path_name)
        
        loaded_model =  TabNetRegressor()
        loaded_model.load_model(saved_filepath)
    
        ### Save OOF for CV ###
        oof_preds[val_idx] += preds_val / len(SEED)
        scores.append(score)
    
        ### Predict on test ###
        model.load_model(saved_filepath)
        preds_test = model.predict(X_test)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))
    
test_preds_all = np.stack(test_cv_preds)


train_features = pd.read_csv('../../inputs/train_features.csv')
train_targets_scored = pd.read_csv('../../inputs/train_targets_scored.csv')
oof = train_features.merge(train_targets_scored, on='sig_id')
oof = oof[train_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

oof[target_cols] = oof_preds
oof[['sig_id']+target_cols].to_csv('oof.csv', index=False)

print(oof.shape)


aucs = []
for task_id in list(train_targets_scored.columns[1:]):
    if task_id == 'sig_id':
        continue
    aucs.append(roc_auc_score(y_true = train_targets_scored[train_features['cp_type']!='ctl_vehicle'].loc[:, task_id],
                              y_score = oof.loc[:,task_id]
                             ))
print(f"Overall AUC: {np.mean(aucs)}")
print(f"Average CV: {np.mean(scores)}")


all_feat = [col for col in df.columns if col not in ["sig_id"]]
# To obtain the same lenght of test_preds_all and submission
test = pd.read_csv("../../inputs/test_features.csv")
sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
tmp["sig_id"] = sig_id

submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
submission.fillna(0, inplace = True)
submission.to_csv("submission.csv", index = None)

print(f"submission.shape: {submission.shape}")

print('done')

