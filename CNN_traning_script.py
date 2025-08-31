import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, AveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Concatenate, GlobalAveragePooling1D, Dense, Dropout, Reshape, Multiply
from keras.regularizers import l2
from sklearn.utils import class_weight, resample
from sklearn.metrics import classification_report
from joblib import dump as save_scaler
from warnings import filterwarnings
filterwarnings("ignore")

# ======== 1. Caricamento e preparazione dataset ==========
# Percorso ai file Parquet (modifica il path a seconda dei tuoi file)
file_paths = glob.glob("*.parquet")
dfs = [pd.read_parquet(f) for f in file_paths]
df = pd.concat(dfs, ignore_index=True)
#unique_values = df["Label"].unique()
#print(unique_values)
print(df.columns)
#df.to_csv("dataset_check.csv")
#print(df.head())
#print(df.tail())
#count = (df["Label"] == "Benign").sum()
#print("Benign count:", count)
#count_div = (df["Label"] != "Benign").sum()
#print("No Benign count:", count_div)

# ======== 2. Pulizia e codifica label ========
df = df.dropna() #<- potrebbe cancellarti il dataset dentro il dataframe, poi ti da errore riga 95 se capita csistema il dataset
#Rimuovi righe incomplete
#count = (df["Label"] == "Benign").sum()
#print("Benign count:", count)
#count_div = (df["Label"] != "Benign").sum()
#print("No Benign count:", count_div)

#Classi che individuano gli attacchi nel dataset
label_col = "Label"  # <-- Nome della colonna con le etichette
label_map = {
    "BENIGN": "0",
    "DDOS ATTACKS-LOIC-HTTP": "1",
    "BOT": "2",
    "SSH-BRUTEFORCE": "3",
    "INFILTERATION": "4",
    "DOS ATTACKS-GOLDENEYE": "5",
    "DOS ATTACKS-SLOWLORIS": "6",
    "BRUTE FORCE -WEB": "7",
    "BRUTE FORCE -XSS": "8",
    "FTP-BRUTEFORCE": "9",
    "SQL INJECTION": "10"
}
df[label_col] = df[label_col].astype(str).str.strip().str.upper()
#print(df[label_col].unique())

# 2. Applica la mappatura
df["y"] = df[label_col].map(label_map)

# 3. Rimuovi righe con etichette non mappate (valori NaN)
df = df[df["y"].notna()]

# 4. Converti la colonna y in interi
df["y"] = df["y"].astype(int)
#print(df[label_col].value_counts(dropna=False))

# REDUCE: concatena tutti i DataFrame validi
#df = pd.concat(mapped_df, ignore_index=True)

# Rimuovi la colonna etichetta testuale
X = df.drop(columns=[label_col, "y"])
y_uni_class = df["y"].unique()
print(y_uni_class) 
y = df["y"].values


# Rendi binario: 0 = BENIGN, 1 = Malicious
df[label_col] = df[label_col].apply(lambda x: 0 if "BENIGN" in x.upper() else 1)

# ======== 3. Separazione feature/target ========
#X = df.drop(columns=[label_col])
y = df[label_col].values
#le = LabelEncoder()
#y = le.fit_transform(df[label_col])
#print(list(le.classes_))
#print(y)


# Rimuovi colonne categoriche se presenti
X = X.select_dtypes(include=["float64", "int64"])
#df_filtered = df[df["Label"] != 0]
#print(df_filtered)
# ======== 4. Standardizzazione ========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== 5. Reshape per CNN ========
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# ======== 6. One-hot encoding ========
y_cat = to_categorical(y, num_classes=2)

# ======== 7. Train/test split ========
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_cat, test_size=0.2, stratify=np.argmax(y_cat, axis=1), random_state=42)

#print("Distribuzione classi nel test set:", np.unique(np.argmax(y_test, axis=1), return_counts=True))

# === 7.1 Bilanciamo train set ===
y_train_int = np.argmax(y_train, axis=1)  # da one-hot a etichette intere
#class_weight_var = class_weight.compute_class_weight(
#    class_weight="balanced",
#    classes=np.unique(y_train_int),
#    y=y_train_int
#)

#class_weights = dict(enumerate(class_weight_var)) # Converti in dizionario per keras
#print("Class weights:", class_weights)

def balance_classes_undersample(X_train, y_train):
    """
    Eseguo undersampling sulla classe BENIGN (0) per bilanciare il dataset.
    Non uso class_weight di sklearn
    Funziona con dati CNN (X a 3 dimensioni) e y one-hot encoded (es. [1, 0] / [0, 1]).
    """
    # Appiattisci X per concatenare con y
    X_flat = X_train.reshape(X_train.shape[0], -1)

    # Unisci X e y in un solo array
    data = np.concatenate((X_flat, y_train), axis=1)

    # Separa le classi: y[:, 0] = BENIGN, y[:, 1] = MALICIOUS
    benign = data[y_train[:, 0] == 1]
    malicious = data[y_train[:, 1] == 1]

    # Sottocampiona la classe BENIGN per pareggiarla ai MALICIOUS
    benign_downsampled = resample(
        benign,
        replace=False,
        n_samples=len(malicious),
        random_state=42
    )

    # Ricombina i dati e mescola come a briscola :D
    balanced_data = np.vstack([benign_downsampled, malicious])
    np.random.shuffle(balanced_data)

    # Dividi di nuovo in X e y
    X_bal = balanced_data[:, :-2].reshape(-1, X_train.shape[1], 1)
    y_bal = balanced_data[:, -2:]

    return X_bal, y_bal

X_train, y_train = balance_classes_undersample(X_train, y_train)

#salviamo lo scaler

save_scaler(scaler, "scaler_pp.pkl")

# ======== 8. Costruzione della CNN ========
def sequence_execitation_block(x, ratio=16):
    prev_layer_dim = x.shape[-1]
    seq = GlobalAveragePooling1D()(x)
    seq = Dense(prev_layer_dim // ratio, activation="elu")(seq)
    seq = Dense(prev_layer_dim, activation="sigmoid")(seq)
    seq = Reshape((1, prev_layer_dim))(seq)
    x = Multiply()([x, seq]) #pesa ogni canale con i valori generati dal blocco di attenzio, prodotto scalare entrata per entrata del vettore
    return x

def residual_block_ultra(x, filters, kernel_sizes=[3,5,7], dropout_rate=0.2):
    shortcut = x
    for k in kernel_sizes:
        x = Conv1D(filters, kernel_size=k, kernel_regularizer=l2(1e-4), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    # Adatta canali per la residual connection se necessario
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, kernel_regularizer=l2(1e-4), padding="same")(shortcut)
    x = Add()([x, shortcut]) #così la rete non peggiora
    
    x = sequence_execitation_block(x)
    
    x = MaxPooling1D()(x)
    x = Dropout(dropout_rate)(x)
    return x

def build_cnn_ultra(input_shape, num_classes=2):
    inputs = Input(shape=input_shape)
    
    # Blocchi residuali
    x = residual_block_ultra(inputs, filters=32, dropout_rate=0.1)
    x = residual_block_ultra(x, filters=64, dropout_rate=0.2)
    x = residual_block_ultra(x, filters=128, dropout_rate=0.25)
    x = residual_block_ultra(x, filters=256, dropout_rate=0.3)
    
    # Pooling globale e fully connected cerchiamo di evitare crollo validation score
    gap1d = GlobalAveragePooling1D()(x)
    gmp1d = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap1d, gmp1d])
    x = Dense(128, kernel_regularizer=l2(1e-4), activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(64, kernel_regularizer=l2(1e-4), activation="relu")(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ======== 9. Addestramento ========
input_shape = (X_train.shape[1], 1)
model = build_cnn_ultra(input_shape)
model.summary()

# ==== 9.1 Evitiamo overfitting ====
#il modello verifica ogni ciclo di addestramento
#aspetta almeno 1 ciclo prima di fare check sui migliormaneti dell"apprendimento
# così facciamo 2 epoche e tanto va bene lo stesso
callbacks_var = [
    EarlyStopping(patience=3, restore_best_weights=True), 
    ReduceLROnPlateau(patience=3, factor=0.5)             
]

history = model.fit(
    X_train, 
    y_train,
    epochs = 4,           
    batch_size = 64,
    validation_data = (X_test, y_test),
#    class_weight = class_weights,
    callbacks = callbacks_var
)

# ======== 10. Valutazione ========
loss, acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(classification_report(y_true_labels, y_pred_labels))


plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======== 11. Salva modello ========
model.save("unclepaskm_cnn_pp.h5")
