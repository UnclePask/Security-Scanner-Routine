#from fastapi import FastAPI
#from pydantic import BaseModel
from tensorflow.keras.models import load_model
#from utils import preprocess_input
from sklearn.preprocessing import StandardScaler
import numpy as np
import sniffing_acion as cattura
import pandas as pd
import glob
from joblib import load as load
# Carica fast API per gui web
#app = FastAPI(title="Attack Detector CNN", version="1.0")


def main():
    model = load_model("unclepaskm_cnn.h5")
    #df = cattura.sniffing_to_file(200).avvio_sniffing()
    ##=====Normalizzazione=====
    #scaler = StandardScaler()
    #categorical_cols = df.columns
    ## 1. Preprocessing dummy
    #df = pd.get_dummies(df, columns=categorical_cols)
    #
    ## 2. Rimuovi Label e converti tutto in float32
    #features = df.drop(columns=['Label'], errors='ignore').astype(np.float32)
    #features_scaled = scaler.fit_transform(features)
    file_paths = glob.glob("*.parquet")
    dfs = [pd.read_parquet(f) for f in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.head(200)
    df = df.drop(columns=['Label'])
    #bisogna trovare un modo per ricalcolare le soglie
    BENIGN_THRESHOLD = 0.7
    MALICIOUS_THRESHOLD = 0.5
    loaded_scaler = load('scaler.pkl')
    pos = pd.DataFrame()
    bnd = pd.DataFrame()
    neg = pd.DataFrame()

    new_normalized_data = loaded_scaler.fit_transform(df.to_numpy())
    for row in new_normalized_data:
        # poi fai predirre la riga dal nostro modello di calssificazione con qualcosa del tipo:
        pippo = row
        input_data = row.reshape(1, -1).astype(np.float32)
        #input_data = np.array(row.values, dtype=np.float32).reshape(1, -1)
        prob = model.predict(input_data)[0][0]  # probabilità classe 0 (BENIGN) nell'entrata colonne della matrice
        #infine classifica la riga con qualcosa di questo tipo
        if prob > BENIGN_THRESHOLD:
            new_line = pd.DataFrame(row)
            pos = pd.concat([pos, new_line], ignore_index = True)
        elif prob < MALICIOUS_THRESHOLD:
            new_line = pd.DataFrame(row)
            neg = pd.concat([neg, new_line], ignore_index = True)
        else:
            new_line = pd.DataFrame(row)
            bnd = pd.concat([bnd, new_line], ignore_index = True)
        
    print(df)
    pos.to_csv("pos.csv")
    neg.to_csv("neg.csv")
    bnd.to_csv("bnd.csv")
    #salva dataframe in CSV
    df.to_csv("test.csv")


#la funzione main deve diventare la funzione acvtion del tasto della nostra interfaccia grafica
#la il numero di pacchetti deve essere inserito da interfaccia grafica
#per funzionare lo sniffing in windows va installato il driver PCAP di NMAP
#se usi un sistema unix-like (ubuntu, debian, suse, bsd o macos) devi lavorare a livello 3 dello strato di rete L3Socket "Chiedi a chat GPT"

if __name__ == "__main__":
    main()
