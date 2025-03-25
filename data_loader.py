import pandas as pd
import json

def load_data_from_csv(args):
    if(args.dataset == 'IMDB_10000'):
        data = pd.read_csv('./data/IMDB_10000.csv')
    elif(args.dataset == 'IMDB_100000'):
        data = pd.read_csv('./data/IMDB_100000.csv')
    return data

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def preprocess_data(data):
    return data
