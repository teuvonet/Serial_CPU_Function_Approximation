import pandas as pd
import numpy as np
import math

maps, inverse_maps, col_maps = {}, {}, {}

def distance_decode(df):
    global maps, inverse_maps, col_maps
    encoded_cols = list(maps.keys())
    for fea in encoded_cols:
        df[fea] = df[col_maps[fea]].astype(str).apply(lambda x: ''.join(x), axis=1)
        df.drop(columns=col_maps[fea],inplace=True)
        df[fea].replace(inverse_maps[fea], inplace=True)
    return df

def distance_encode(df):
    global maps, inverse_maps, col_maps
    obj_headers = df.select_dtypes(include=['object']).columns
    i = 0

    while i < len(obj_headers):
        try:
            df[obj_headers[i]] = pd.to_datetime(df[obj_headers[i]])
            print("%s is datetime" % obj_headers[i])
            del obj_headers[i]
        except ValueError:
            # print("%s not datetime" % obj_headers[i])
            i += 1
    for fea in obj_headers:
        encode_map = {}
        decode_map = {}
        cla = df[fea].unique()
        num_digits = int(math.ceil(math.log(len(cla),2)))
        new_col_names = []
        for k in range(num_digits):
            new_col_names.append(fea+"_"+str(k))
        col_maps[fea] = new_col_names
        for i in range(len(cla)):
            encode_map[cla[i]] = bin(i)[2:].zfill(num_digits)
            decode_map[bin(i)[2:].zfill(num_digits)] = cla[i]
        # print("map for %s is %s" % (fea, encode_map))
        maps[fea] = encode_map
        inverse_maps[fea] = decode_map

        col_names = col_maps[fea]
        df[fea].replace(maps[fea],inplace=True)
        # print(df[fea])
        for i in range(len(col_names)):
            df[col_names[i]] = df[fea].map(lambda x: x[i]).astype(np.int32)
        df.drop(columns=[fea], inplace=True)
    return df

if __name__ == '__main__':
    dataset_name = 'House_Price_Adv_Regression'
    df = pd.read_csv('Datasets/'+dataset_name+'/Train/'+dataset_name+'_Train_seed0.csv')#.drop(['Unnamed: 0'],axis=1)
    df = distance_encode(df)
    print(df.columns)
    df = distance_decode(df)