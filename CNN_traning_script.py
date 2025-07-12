import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, BatchNormalization, GlobalMaxPooling1D
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# ======== 1. Caricamento e preparazione dataset ==========


# Percorso ai file Parquet (modifica il path a seconda dei tuoi file)
file_paths = glob.glob("*.parquet")
dfs = [pd.read_parquet(f) for f in file_paths]
df = pd.concat(dfs, ignore_index=True)
#unique_values = df['Label'].unique()
#print(unique_values)
#df.to_csv('dataset_check.csv')
#print(df.head())
#print(df.tail())
#count = (df['Label'] == 'Benign').sum()
#print("Benign count:", count)
#count_div = (df['Label'] != 'Benign').sum()
#print("No Benign count:", count_div)

# ======== 2. Pulizia e codifica label ========
df = df.dropna() #<- potrebbe cancellarti il dataset dentro il dataframe, poi ti da errore riga 95 se capita csistema il dataset
#Rimuovi righe incomplete
#count = (df['Label'] == 'Benign').sum()
#print("Benign count:", count)
#count_div = (df['Label'] != 'Benign').sum()
#print("No Benign count:", count_div)

#Classi che individuano gli attacchi nel dataset
label_col = 'Label'  # <-- Nome della colonna con le etichette
label_map = {
    'BENIGN': '0',
    'DDOS ATTACKS-LOIC-HTTP': '1',
    'BOT': '2',
    'SSH-BRUTEFORCE': '3',
    'INFILTERATION': '4',
    'DOS ATTACKS-GOLDENEYE': '5',
    'DOS ATTACKS-SLOWLORIS': '6',
    'BRUTE FORCE -WEB': '7',
    'BRUTE FORCE -XSS': '8',
    'FTP-BRUTEFORCE': '9',
    'SQL INJECTION': '10'
}
df[label_col] = df[label_col].astype(str).str.strip().str.upper()
#print(df[label_col].unique())

# 2. Applica la mappatura
df['y'] = df[label_col].map(label_map)

# 3. Rimuovi righe con etichette non mappate (valori NaN)
df = df[df['y'].notna()]

# 4. Converti la colonna y in interi
df['y'] = df['y'].astype(int)
#print(df[label_col].value_counts(dropna=False))

# REDUCE: concatena tutti i DataFrame validi
#df = pd.concat(mapped_df, ignore_index=True)

# Rimuovi la colonna etichetta testuale
X = df.drop(columns=[label_col, 'y'])
y_uni_class = df['y'].unique()
print(y_uni_class) 
y = df['y'].values


# Rendi binario: 0 = BENIGN, 1 = Malicious
#df[label_col] = df[label_col].apply(lambda x: 0 if "BENIGN" in x.upper() else 1)

# ======== 3. Separazione feature/target ========
#X = df.drop(columns=[label_col])
#y = df[label_col].values
#le = LabelEncoder()
#y = le.fit_transform(df[label_col])
#print(list(le.classes_))
#print(y)


# Rimuovi colonne categoriche se presenti
X = X.select_dtypes(include=['float64', 'int64'])
#df_filtered = df[df['Label'] != 0]
#print(df_filtered)
# ======== 4. Standardizzazione ========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======== 5. Reshape per CNN ========
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# ======== 6. One-hot encoding ========
y_cat = to_categorical(y, num_classes=11)

# ======== 7. Train/test split ========
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_cat, test_size=0.2, stratify=np.argmax(y_cat, axis=1), random_state=42)

#print("Distribuzione classi nel test set:", np.unique(np.argmax(y_test, axis=1), return_counts=True))

# === 7.1 Bilanciamo train set ===
y_train_int = np.argmax(y_train, axis=1)  # da one-hot a etichette intere
class_weight_var = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)

class_weights = dict(enumerate(class_weight_var)) # Converti in dizionario per keras
print("Class weights:", class_weights)

# ======== 8. Costruzione della CNN ========
def build_cnn_model(input_shape, num_classes=11):
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Dropout(0.3))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ======== 9. Addestramento ========
input_shape = (X_train.shape[1], 1)
model = build_cnn_model(input_shape)
model.summary()

# ==== 9.1 Evitiamo overfitting ====
#il modello verifica ogni ciclo di addestramento
#aspetta almeno 1 ciclo prima di fare check sui migliormaneti dell'apprendimento
# così facciamo 2 epoche e tanto va bene lo stesso
callbacks_var = [
    EarlyStopping(patience=1, restore_best_weights=True), 
    ReduceLROnPlateau(patience=1, factor=0.5)             
]

history = model.fit(
    X_train, 
    y_train,
    epochs = 2,           
    batch_size = 64,
    validation_data = (X_test, y_test),
    class_weight = class_weights,
    callbacks = callbacks_var
)

# ======== 10. Valutazione ========
loss, acc = model.evaluate(X_test, y_test)
#print(f"Test Accuracy: {acc:.4f}")
#print(f"Distribuzione classi:")
#print(np.sum(y, axis=0))
#print(X.shape)
#print(y[:10])
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(classification_report(y_true_labels, y_pred_labels))


plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======== 11. Salva modello ========
model.save("unclepaskm_cnn.h5")

# 2 epoche:

# 2 classes
#               precision    recall  f1-score   support
# 
#            0       0.98      0.70      0.82    701117
#            1       0.45      0.94      0.61    184421 <- più del 50% di falsi allarmi
#                                                          che dovranno andare in BND
#     accuracy                           0.75    885538
#    macro avg       0.72      0.82      0.72    885538
# weighted avg       0.87      0.75      0.78    885538

# 11 classes
#               precision    recall  f1-score   support
# 
#            0       0.96      0.03      0.06    899825
#            1       0.14      0.93      0.24    115073
#            2       0.00      0.00      0.00     28907
#            3       0.00      0.00      0.00     18810
#            4       0.03      0.06      0.04     23697
#            5       0.02      0.58      0.04      8281
#            6       0.18      0.71      0.29      1982
#            7       0.01      0.37      0.02        68
#            8       0.00      0.00      0.00        30
#            9       0.00      0.00      0.00        10
#           10       0.00      0.00      0.00        10
# 
#     accuracy                           0.13   1096693
#    macro avg       0.12      0.24      0.06   1096693
# weighted avg       0.80      0.13      0.07   1096693
