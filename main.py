'''
Created on 4 ago 2025

@author: pasquale
'''
import pandas as pd
import numpy as np
import glob
import math
from staticMethodsCollection import staticMethods as sm
import argparse
from keras.models import load_model
from joblib import load as load

def entropy(prob):
    return -sum([p * math.log2(p) for p in prob if p > 0])

def load_dataset(dataset_type: str, limit: int):
    if dataset_type == "all" or dataset_type == "debug":
        file_paths = glob.glob("*.parquet")
    else:
        file_paths = dataset_type
    dfs = [pd.read_parquet(f) for f in file_paths] 
    df = pd.concat(dfs)
    if dataset_type == "debug":
        df = df.head(limit)
    df.to_csv("original_data.csv")
    df = df.drop(columns=['Label'])
    return df

def read_model_and_normalization(path_model: str, path_scaler: str, data_frame: pd.DataFrame()):
    model = load_model(path_model, compile = False)
    scaler = load(path_scaler)
    df = data_frame.select_dtypes(include=['float64', 'int64'])
    normalized_data = scaler.fit_transform(df.to_numpy())
    return model, normalized_data

def predict_as_batch(model, normalized_data):
    normalized_data = normalized_data.astype(np.float32)
    #reshape per CNN se 1D: (none, 16, 1) se 2D: (none, 16, 1, 1) ............ (77, 1) .... ma tu c'è fatt e pall tant fratèèèèèèèèè ma come cazz pigli 32 77???? e bast!!!!!!"!
    normalized_data_reshaped = normalized_data.reshape(normalized_data.shape[0], normalized_data.shape[1], 1)
    probs_predicted = model.predict(normalized_data_reshaped)
    return probs_predicted

def merge_with_predicted_values(df: pd.DataFrame(), probs, entropies):
    predicted = df.copy()
    predicted["prob_ben"] = probs[:,0]
    predicted["prob_mal"] = probs[:,1]
    predicted["entropy"] = entropies
    return predicted

def calc_threshold(entropies_array, predicted_data_frame: pd.DataFrame()):
    print(f"\nCalcolo delle soglie:\n")
    array_H = np.array(entropies_array)
    beta = np.percentile(array_H, 25)
    alpha = np.percentile(array_H, 75)
    predicted_data_frame["margine"] = (predicted_data_frame["prob_ben"] - predicted_data_frame["prob_mal"]).abs()
    #    condizione di frontiera per alpha: (predicted["entropy"] <= alpha) | (margin >= quartile)
    print(f"\nalpha = {alpha} \nbeta = {beta}")
    return alpha, beta, predicted_data_frame

def genera_roughset(threshold: str, alpha, beta, c_df: pd.DataFrame()):
    print(f"\nTipo valore della soglia: {threshold} \nGenerazione RoughSet In corso ...\n")
    try:
        if threshold == "margine":
            roughset_pos = (c_df["prob_ben"] > c_df["prob_mal"]) & (beta <= c_df["margine"]) & (c_df["margine"] < alpha)
            roughset_neg = (c_df["prob_ben"] < c_df["prob_mal"]) & (beta <= c_df["margine"]) & (c_df["margine"] < alpha)
            roughset_bnd = ~(roughset_pos | roughset_neg)
        elif threshold == "tollerante":
            roughset_pos = (c_df["prob_ben"] > c_df["prob_mal"]) & (c_df["margine"] > beta)
            roughset_neg = (c_df["prob_ben"] < c_df["prob_mal"]) & (c_df["margine"] > beta)
            roughset_bnd = ~(roughset_pos | roughset_neg)
        elif threshold == "entropia":
            roughset_pos = (c_df["prob_ben"] > c_df["prob_mal"]) & (c_df["entropy"] < beta)
            roughset_neg = (c_df["prob_ben"] < c_df["prob_mal"]) & (c_df["entropy"] < beta)
            roughset_bnd = ~(roughset_pos | roughset_neg)
        else:
            print(f"\nErrore (1): Specificare il tipo di utilizzo della soglia per generare i RoughSet!")
        
        pos = c_df[roughset_pos].copy()
        neg = c_df[roughset_neg].copy()
        bnd = c_df[roughset_bnd].copy()
        print(f"Cardinalità di Pos = {pos.index.size}")
        print(f"Cardinalità di Bnd = {bnd.index.size}")
        print(f"Cardinalità di Neg = {neg.index.size}")
        pos.to_csv("pos.csv")
        neg.to_csv("neg.csv")
        bnd.to_csv("bnd.csv")
        print("\nInfo: RoughSet Generati!")
        return pos, neg, bnd
    
    except:
        print(f"\nErrore (2): RoughSet non generati, vengono restituiti i data frame vuoti!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def twd_pipeline(data_frame: pd.DataFrame(), path_model: str, path_scaler: str, type_threshold: str):
    model, normalized_data = read_model_and_normalization(path_model, path_scaler, data_frame)
    
    probs = predict_as_batch(model, normalized_data)
    entropies = np.array([entropy(p) for p in probs])
    predicted = merge_with_predicted_values(data_frame, probs, entropies)
    
    alpha, beta, predicted = calc_threshold(entropies_array = entropies, 
                   predicted_data_frame = predicted)
    
    pos, neg, bnd = genera_roughset(threshold = type_threshold, 
                                    alpha = alpha, 
                                    beta = beta, 
                                    c_df = predicted)
    
    return pos, neg, bnd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="debug or all")
    parser.add_argument("--limit", type=int, default=10, help="Numero di record da testare nel dataframe")
    parser.add_argument("--epoch", type=int, default=10, help="Numero di epoche per la boundary area")
    args = parser.parse_args()
    epoche = args.epoch
    df = load_dataset(args.type, args.limit)
    
    
    pos_our, neg_our, bnd_our = twd_pipeline(data_frame = df, 
                                             path_model = "unclepaskm_cnn_multi.keras", 
                                             path_scaler = "scaler_multi.pkl", 
                                             type_threshold = "entropia")
    sm.plotRoughSet(pos_our, neg_our, bnd_our, "CNN Convoluzionale UNISA - 1° lancio")
    
    i= 0
    pos_our_list = pd.DataFrame()
    neg_our_list = pd.DataFrame()
    while i < epoche:
        bnd_our = bnd_our.drop(columns=["prob_ben", "prob_mal", "entropy"])
        pos_our1, neg_our1, bnd_our = twd_pipeline(data_frame = bnd_our, 
                                             path_model = "unclepaskm_cnn_multi.keras", 
                                             path_scaler = "scaler_multi.pkl", 
                                             type_threshold = "entropia")
        pos_our_list = pd.concat([pos_our_list, pos_our1])
        neg_our_list = pd.concat([neg_our_list, neg_our1])
        i = i+1
    
    pos_our = pd.concat([pos_our, pos_our_list])
    neg_our = pd.concat([neg_our, neg_our_list])
    sm.plotRoughSet(pos_our, neg_our, bnd_our, "CNN Convoluzionale UNISA - Boundary Area Analyzed")
    
    pos_mono, neg_mono, bnd_mono = twd_pipeline(data_frame = df, 
                                             path_model = "unclepaskm_cnn_mono.keras", 
                                             path_scaler = "scaler_mono.pkl", 
                                             type_threshold = "entropia")
    sm.plotRoughSet(pos_mono, neg_mono, bnd_mono, "CNN Convoluzionale 3x3 Kernel - 1° lancio")
    
    j= 0
    pos_mono_list = pd.DataFrame()
    neg_mono_list = pd.DataFrame()
    while j < epoche:
        bnd_mono = bnd_mono.drop(columns=["prob_ben", "prob_mal", "entropy"])
        pos_mono1, neg_mono1, bnd_mono = twd_pipeline(data_frame = bnd_mono, 
                                             path_model = "unclepaskm_cnn_mono.keras", 
                                             path_scaler = "scaler_mono.pkl", 
                                             type_threshold = "entropia")
        pos_our_list = pd.concat([pos_mono_list, pos_mono1])
        neg_our_list = pd.concat([neg_mono_list, neg_mono1])
        j = j+1
    
    pos_mono = pd.concat([pos_mono, pos_mono_list])
    neg_mono = pd.concat([neg_mono, neg_mono_list])
    sm.plotRoughSet(pos_mono, neg_mono, bnd_mono, "CNN Convoluzionale 3x3 Kernel - Boundary Area Analyzed")
    
    print(f"Cardinalità di PosUnisa = {pos_our.index.size}")
    print(f"Cardinalità di BndUnisa = {bnd_our.index.size}")
    print(f"Cardinalità di NegUnisa = {neg_our.index.size}")
    print(f"Cardinalità di PosMono = {pos_mono.index.size}")
    print(f"Cardinalità di BndMono = {bnd_mono.index.size}")
    print(f"Cardinalità di NegMono = {neg_mono.index.size}")
    
     
if __name__ == "__main__":
    main()