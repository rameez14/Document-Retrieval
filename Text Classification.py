import time
import os
import re
import collections
import numpy as np
import argparse


# read the files in a list
def reading_f(folder):
    doc = os.listdir(folder)
    data_f = []
    for doc in doc:
        with open(folder+"/"+doc) as f:
            data_f.append(f.read())

    return data_f

# Going through all the relevant files in the list by extracting
def derive (f_d, syb,sign):
    characteristic = []
    keys = []
    f_1,k_1 = unigram(f_d, syb, sign)
    characteristic.append(f_1)
    keys.append(k_1)
    return characteristic, keys

#Unigram and getting all the relevant features
def unigram(f_d, sym, sign):
    characteristic = []
    unique_k = []
    for f_d in f_d:
        dict_d = dict(collections.Counter(re.sub("[^\w']"," ",f_d).split()))
        dict_d[sym] = sign
        characteristic.append(dict_d)
        unique_k.extend(dict_d.keys())

    return characteristic,unique_k

# Building the matrix and getting all the relevant keys
def mx(info, k):
    m = np.zeros((len(info), len(k)))
    indexing = 0
    for word in k:
        for index in range(len(info)):
            if word in info[index]:
                m[index][indexing] = info[index][word]
        indexing +=1
    return m

# Dividing all the relevant data into the training set and the valid set
def dividing(score, num_gen):
    positive_class = int(num_gen / 2)
    negitive_class = int(num_gen - positive_class)

    val_idx = np.append(np.nonzero(score[:,0]>0)[0][:positive_class],np.nonzero(score[:,0]<0)[0][:negitive_class])
    val_d = score[val_idx]
    train_d = np.delete(score, val_idx, axis=0)

    return train_d,val_d

# Testing the training weight
def training_weight (w, val_input):
    pos_d = val_input[np.nonzero(val_input[:,0]>0)[0]]
    neg_d = val_input[np.nonzero(val_input[:,0]<0)[0]]
    is_pos = np.sign(np.dot(np.delete(pos_d, [0], axis=1), w))
    is_neg = np.sign(np.dot(np.delete(neg_d, [0], axis=1), w))

    pos_t = (is_pos>0).sum()
    f_pos = pos_d.shape[0] - pos_t
    t_neg = (is_neg < 0).sum()
    f_neg = neg_d.shape[0] - t_neg

    precision = pos_t/(pos_t+f_pos)
    recall = pos_t/(pos_t + f_neg)

    return pos_t,t_neg,precision,recall

# main
start = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help="folder", default="review_polarity")
args = parser.parse_args()
folder_name = args.folder if 'folder' in args else 'review_polarity'
folder_name += "/txt_sentoken"
pos_folder = folder_name+"/pos"
neg_folder = folder_name+"/neg"


positive = 1
negitive = -1
token = "sign"
val_number = 200

positive_documents = reading_f(pos_folder)
negitive_documents = reading_f(neg_folder)

print('Reading the relevant files: %.2fs'%(time.perf_counter()-start))
start = time.perf_counter()

positive_features, positive_k = derive(positive_documents, token, positive)
negitive_features, negitive_k = derive(negitive_documents, token, negitive)
unique_ks = np.unique(np.append(positive_k[0],negitive_k[0]))
k = np.nonzero(unique_ks == token)[0][0]
unique_ks[[0, k]] = unique_ks[[k, 0]]

# Time taken ti extract relevant characteristics
print('Time to extract relevant characteristics: %.2fs '%(time.perf_counter()-start))
start = time.perf_counter()

dm = mx(np.append(positive_features, negitive_features), unique_ks)
print('Time to build the matrix: %.2fs'%(time.perf_counter()-start))
start = time.perf_counter()

dm = np.random.permutation(dm)
train_d, val_d = dividing(dm, val_number)
print('Time of dividing training and valid data: %.2fs'%(time.perf_counter()-start))
start = time.perf_counter()

# iteration stages
iter = train_d[:,0].reshape((train_d.shape[0],1))
train_d = np.delete(train_d,[0],axis=1)
w = np.zeros((train_d.shape[1], 1))
Maximum_iterations = 10
er = train_d.shape[0]

print('Begin training')


while(er>=200):
    previous = np.sign(np.dot(train_d, w))
    idx = np.nonzero(previous != iter)[0]
    vector = np.zeros((train_d.shape[0],1))
    vector[idx] = iter[idx]
    acc = np.dot(train_d.T, vector) / train_d.shape[0]
    w += acc
    er = (np.absolute(iter - previous)).sum()
    if (k+1)%50==0:
     print("Iteration %d : Error  %d  " % (k+1, er))

print('Time to train: %.2f s'%(time.perf_counter()-start))
t_pos,t_neg,precision,recall = training_weight(w, val_d)
print("Positive: %d \n Negative: %d \n Precision: %.2f \n Recall: %.2f \n" % (t_pos, t_neg, precision, recall))
