# -*- coding: utf-8 -*-
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from nltk.stem.snowball import SnowballStemmer
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from datasets import load_dataset
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from string import punctuation
from tqdm import tqdm
from pymystem3 import Mystem
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import openpyxl
import random
import torch
import pylab
import nltk
import umap
import boto3
import json
import sys
import re
import os
from pymongo import MongoClient
from urllib.parse import quote_plus as quote
from bson.objectid import ObjectId

nltk.data.path.append('/app/nltk_data')
warnings.filterwarnings("ignore")

class Preprocessor:
    def __init__(self, max_len=128, batch_size=32, model_name="bert-base-uncased"):
        self.batch_size = batch_size
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = lambda x: tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors='pt')

    def prepare_data(self, text, label):
        text = list(map(self.tokenizer, text))
        label = list(map(int, label))
        num_clusters = len(set(label))
        label = torch.tensor(label)
        for i in range(len(text)):
            text[i]["input_ids"] = text[i]["input_ids"].squeeze(0)
            text[i]["token_type_ids"] = text[i]["token_type_ids"].squeeze(0)
            text[i]["attention_mask"] = text[i]["attention_mask"].squeeze(0)
            text[i]["label"] = label[i]
        shuffled_dataloader = DataLoader(text, batch_size=self.batch_size, shuffle=True)
        unshuffled_dataloader = DataLoader(text, batch_size=self.batch_size, shuffle=False)
        return shuffled_dataloader, unshuffled_dataloader, num_clusters

class DatasetAutoEnc(Dataset):
    def __init__(self, text):
        super().__init__()
        self.text = text
    def __getitem__(self, idx):
        return self.text['input_ids'][idx], self.text['attention_mask'][idx], self.text['token_type_ids'][idx]

    def __len__(self):
        return len(self.text['attention_mask'])

def batch_collate(batch):
    input_ids, attention_mask, label = torch.utils.data._utils.collate.default_collate(batch)
    max_length = attention_mask.sum(dim=1).max().item()
    attention_mask, input_ids = attention_mask[:, :max_length], input_ids[:, :max_length]
    return input_ids, attention_mask, label

class Plot_data():
    def __init__(self, embeddings, classes):
        self.emb = embeddings
        self.classes = classes
    def umap_data(self, embeddings, n) -> np.array:
        ''' Dimension reduction function '''
        reducer = umap.UMAP(n_components=n)
        scaled_data = StandardScaler().fit_transform(embeddings)
        embedding = reducer.fit_transform(scaled_data)
        return embedding
    def centers(self, umap_data, labels):
        ln = len(np.unique(list(labels)))
        if -1 in np.unique(labels):
            ln -= 1
        centers = np.empty(shape=(len(np.unique(labels))-1, 4))
        for ind, class_ in enumerate(np.unique(labels)):
            x = 0
            y = 0
            i = 0
            for ind, elem in enumerate(umap_data):
                if labels[ind] == class_:
                    x += elem[0]
                    y += elem[1]
                    i += 1
            if class_ != -1 or class_ < len(centers):centers[class_] = [x/i, y/i, np.sqrt(i/len(labels)*200000), class_]
        centers[:,0] += abs(min(centers[:,0]))
        centers[:,1] += abs(min(centers[:,1]))
        return centers
    def plt_data(emb, classes):
        umap_data = self.umap_data(emb, 2)
        arr = np.empty(shape=(len(umap_data, 3)))
        for ind, elem in enumerate(umap_data):
            arr[ind] = [elem, class_[ind]]
        return arr

class Main:
    def __init__(self, data, language='russian'):
        self.punctuation = punctuation + '»' + '«'
        self.lang = language
        self.download = True
        try:
            get_object_response = S3_CLT.get_object(
                Bucket=S3_BUCKET,
                Key=data
            )
            self.data = vtb = pd.read_excel(
                get_object_response['Body'].read(),
                engine='openpyxl')['text'].tolist()[:200]
        except Exception as e:
            print('ERROR', e)
            self.download = False
        if language == 'russian':
            self.stem = Mystem()
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
            self.model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        else:
            self.stem = SnowballStemmer(language='english')
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = stopwords.words(language)
        if language == 'english':
            self.stop_words += ['i']

    def mean_pooling(self, model_output, attention_mask) -> np.array:
        ''' Func for getting text embeddings '''
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_text_emb(self, data, tokenizer, model) -> np.array:
        ''' Func to get embeddigs with DeepPavlov model '''
        arr = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        encoded_input = DatasetAutoEnc(
            tokenizer(data, 
                      padding='max_length', 
                      truncation=True, 
                      max_length=128, return_tensors='pt'))
        loader = DataLoader(encoded_input, batch_size=100, shuffle=True)#, collate_fn=batch_collate)
        with torch.no_grad():
            with tqdm(total=len(loader)) as tq:
                for i, batch in enumerate(loader):
                    input_, mask, type_ = batch
                    output = model(input_.to(device), mask.to(device), type_.to(device))
                    sentence_embeddings = self.mean_pooling(output, mask.to(device))
                    arr.append(sentence_embeddings.cpu().detach().numpy())
                    tq.update(1)
            arr = np.concatenate(arr)
        return arr

    def umap_data(self, embeddings, n) -> np.array:
        ''' Dimension reduction function '''
        reducer = umap.UMAP(n_components=n)
        scaled_data = StandardScaler().fit_transform(embeddings)
        embedding = reducer.fit_transform(scaled_data)
        return embedding

    def remove_nan(self, data):
        for ind, x in enumerate(data):
            if type(x) != type("a"):
                data.pop(ind)

    def remove_urls(self, documents) -> list:
        ''' Remove urls from strings in list '''
        return [re.sub('https?:\/\/.*?[\s+]', '', " "+text+" ") for text in documents]

    def tokenize(self, data) -> list:
        ''' Tokenization '''
        return [nltk.word_tokenize(sentence) \
                            for sentence in data]

    def to_lower(self, data) -> list:
        ''' lower() '''
        return [[token.lower() for token in token_sentence] \
                    for token_sentence in data]

    def without_punct(self, data) -> list:
        ''' Delete punctuation '''
        return [[token for token in token_sentence if token not in punctuation] \
                    for token_sentence in data]

    def without_nums(self, data) -> list:
        ''' Delete nums '''
        return [[token for token in token_sentence if token.isalpha()] \
                    for token_sentence in data]

    def without_stop_words(self, data) -> list:
        ''' Delete stopwords '''
        return [[token for token in token_sentence if token not in self.stop_words] \
                    for token_sentence in data]

    def lemmatization(self, data) -> list:
        ''' Tokens lemmatizations '''
        if self.lang == 'russian':
            return [self.stem.lemmatize(token)
                    for token in data]
        else:
            return [[self.stem.stem(token) for token in token_sentence]
                    for token_sentence in data]

    def to_texts(self, data) -> list:
        ''' Convertation tokens to texts '''
        arr = []
        for text in data:
            txt = ''
            for word in text:
                txt += (word + ' ')
            arr.append(txt.replace('\n', ''))
        return arr

    def preprocessing(self, data, if_to_txt=True) -> list:
        ''' Text preprocessing '''
        if self.lang == 'russian':
            self.remove_nan(data)
            data = self.remove_urls(data)
            data = list(map(lambda x: re.sub(r'\W+', ' ', x.replace('_', '')), data))
            data = self.tokenize(data)
            data = self.to_lower(data)
            data = self.without_punct(data)
            data = self.without_nums(data)
            data = self.without_stop_words(data)
            data = self.to_texts(data)
            data = self.lemmatization(data)
            data = self.without_stop_words(data)
            if if_to_txt: data = self.to_texts(data)
        else:
            self.remove_nan(data)
            data = self.tokenize(data)
            data = self.to_lower(data)
            data = self.without_punct(data)
            data = self.without_nums(data)
            data = self.without_stop_words(data)
            data = self.lemmatization(data)
            data = self.without_stop_words(data)
            if if_to_txt: data = self.to_texts(data)
        return data

    def idf(self, dict_, vectorizer) -> list:
        arr_idf = []
        # Filling array with IDF
        for class_, big in dict_.items():
            if class_ != -1:
                X_tfidf = vectorizer.fit(big)
                idf = pd.DataFrame(X_tfidf.get_feature_names(), columns=['token'])
                idf['idf'] = X_tfidf.idf_
                arr_idf.append([class_, idf.sort_values(['idf'])[:20], len(idf)])
        return arr_idf

    def tf_idf(self, sk_corpus, dict_, vectorizer):
        arr_tfidf = []
        # Filling array with TF-IDF
        X_tfidf = vectorizer.fit_transform(dict_.values())
        ind_arr = [list(np.argsort(-x)) for x in X_tfidf.toarray()]
        for class_, cluster in enumerate(ind_arr):
            wl = []
            for word in cluster:
                if word < len(sk_corpus[class_]):
                    sk_word = sk_corpus[class_][int(word)]
                    if sk_word != " " and not '\n' in sk_word: wl.append(sk_word) 
            idf = pd.DataFrame(wl, columns=['token'])
            arr_tfidf.append([class_, idf[:20]])
        return arr_tfidf

    def tf_idf_both(self, data, classes):

        # Arrays to return
        dict_ = {}

        # Preparing dicts
        for label in np.unique(classes):
            if label != -1:
                dict_[label] = []

        # Filling dict for IDF
        for ind, class_ in enumerate(classes):
            if class_ != -1:
                dict_[class_].append(" "+norm_data[ind]+" ")

        # IDF
        arr_idf = self.idf(dict_, vectorizer)

        # Preparing array for TF-IDF
        for class_, big in dict_.items():
            dict_[class_] = ''.join(big)

        arr_tfidf = self.tf_idf(sk_corpus, dict_, vectorizer)

        return arr_idf, arr_tfidf

    def plot_data(self, embeddings, class_, umap_data):
        arr = []
        for ind, elem in enumerate(umap_data):
            arr.append([elem, class_[ind]])
        return arr

    def get_json(self, i_map, arr_tfidf, dict_):
        answer = {"status":"success","payload":{"intertopic_map":[],"topics":[],"documents":[]}}

        if not self.download:
            answer = {
                "status":"error",
                "project_id":1
            }
            return answer

        for ind, dot in enumerate(i_map):
            answer["payload"]["intertopic_map"].append(
                {
                "id":int(dot[3]),
                "keywords":list(arr_tfidf[ind][1]['token'])[:5],
                "size":dot[2],
                "cord_x":dot[0],
                "cord_y":dot[1]
            })
        for ind, dot in enumerate(i_map):
            answer["payload"]["topics"].append(
                {
                "id":int(dot[3]),
                "keywords":list(arr_tfidf[ind][1]['token'])[:5],
                "timestamps":[
                    {
                    "timestamp":None,
                    "value":dot[2],
                    "keywords":list(arr_tfidf[ind][1]['token'])[:5]
                    }
                ]
            })
        for ind, dot in enumerate(i_map):
            answer["payload"]["documents"].append(
            {
            "id":ind,
            "cord_x":dot[0],
            "cord_y":dot[1],
            "cluster_id":dot[3],
            "keywords": [x for x in dict_[ind][0].split(" ")[:5] if x != '']
            }
            )
        return answer

    def predict(self):
        '''
        Main function

        '''
        ############################################################
        #----------------------Preparing data----------------------#
        ############################################################

        with tqdm(total=7) as tq:
            sk_corpus = [
                [x for x in sublist if x != ' ']
                for sublist in self.preprocessing(self.data, False)
            ]; tq.update(1)
            norm_data = self.to_texts(sk_corpus); tq.update(1)
            if self.lang == 'russian':emb_data = self.get_text_emb(
                norm_data,
                self.tokenizer,
                self.model
            ); tq.update(1)
            else: emb_data = self.model.encode(norm_data)
            umap_data = self.umap_data(emb_data, 3); tq.update(1)
            #PCA(n_components=).fit_transform(emb_data)#self.umap_data(emb_data, 3); tq.update(1)
            token = [token for sublist in sk_corpus for token in sublist]; tq.update(1)
            freq_tokens = Counter(token); tq.update(1)
            vectorizer = TfidfVectorizer(vocabulary=list(freq_tokens.keys())); tq.update(1)

        ############################################################
        #--------------------------TF-IDF--------------------------#
        ############################################################

        dbs = DBSCAN().fit(umap_data)
        plt_data = self.plot_data(
            emb_data,
            dbs.labels_,
            PCA(n_components=2).fit_transform(emb_data)
        )

        labels = dbs.labels_

        plot_data_class = Plot_data(emb_data, labels)

        # Arrays to return
        dict_ = {}

        # Preparing dicts
        for label in np.unique(labels):
            if label != -1:
                dict_[label] = []

        # Filling dict for IDF
        for ind, class_ in enumerate(labels):
            if class_ != -1:
                dict_[class_].append(" "+norm_data[ind]+" ")
        dict_for_json = dict_.copy()
        # IDF
        arr_idf = self.idf(dict_, vectorizer)

        # Preparing array for TF-IDF
        for class_, big in dict_.items():
            dict_[class_] = ''.join(big)

        arr_tfidf = self.tf_idf(sk_corpus, dict_, vectorizer)

        # Intertopic map
        i_map = plot_data_class.centers(umap_data, labels)

        answer = self.get_json(i_map, arr_tfidf, dict_for_json)

        return answer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇 alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ----- S3 credentials -----
S3_BUCKET = os.environ['S3_BUCKET']
SERVICE_NAME = os.environ['SERVICE_NAME']
KEY = os.environ['KEY']
SECRET = os.environ['SECRET']
ENDPOINT = os.environ['ENDPOINT']
SESSION = boto3.session.Session()
S3_CLT = SESSION.client(
    service_name=SERVICE_NAME,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    endpoint_url=ENDPOINT
)
S3_RES = SESSION.resource(
    service_name=SERVICE_NAME,
    aws_access_key_id=KEY,
    aws_secret_access_key=SECRET,
    endpoint_url=ENDPOINT
)
# ----- MongoDB credentials -----
MONGODB_HOST = os.environ['MONGODB_HOST']
MONGODB_DATABASE = os.environ['MONGODB_DATABASE']
MONGODB_USERNAME = os.environ['MONGODB_USERNAME']
MONGODB_PASSWORD = os.environ['MONGODB_PASSWORD']

def main_dbs():
    input_data = dict(
        prj_id = sys.argv[1],
        file_name_input = sys.argv[2],
    )
    print(input_data)

    main = Main(data=input_data['file_name_input'])
    info = main.predict()
    info = json.loads(json.dumps(info, cls=NpEncoder))

    url = 'mongodb://{user}:{pw}@{hosts}/?replicaSet={rs}&authSource={auth_src}'.format(
        user=quote(MONGODB_USERNAME),
        pw=quote(MONGODB_PASSWORD),
        hosts=','.join([MONGODB_HOST]),
        rs='rs01',
        auth_src=MONGODB_DATABASE
    )
    db = MongoClient(url, tlsCAFile='/app/CA.pem')['classify']
    filter = { '_id': ObjectId(input_data['prj_id']) } 
    newValue = { "$set": { 'dbs_payload': info } }
    db['projects'].update_one(filter, newValue)
    print('saved to database')

if __name__ == "__main__":
    main_dbs()
