import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from staticMethodsCollection import staticMethods as sm
from joblib import dump as save_scaler
from warnings import filterwarnings
from cnnMonokernel import cnnMonokernel
filterwarnings("ignore")

# 1. Carichiamo dataset ed individuiamo classi e colonna delle realizzazioni
df = sm.readDatasetParquet("*.parquet")

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

# 2. Applica la mappatura
df["y"] = df[label_col].map(label_map)

# 3. Rimuovi righe con etichette non mappate (valori NaN)
df = df[df["y"].notna()]

# 4. Converti la colonna y in interi
df["y"] = df["y"].astype(int)
#print(df[label_col].value_counts(dropna=False))
# REDUCE: concatena tutti i DataFrame validi
#df = pd.concat(mapped_df, ignore_index=True)

# 5. Rimuovi la colonna etichetta testuale
X = df.drop(columns=[label_col, "y"])
y_uni_class = df["y"].unique()
y = df["y"].values

# 6. Rendi binario il dataset: 0 = BENIGN, 1 = Malicious
df[label_col] = df[label_col].apply(lambda x: 0 if "BENIGN" in x.upper() else 1)
y = df[label_col].values
#da decommentare per scriptare più classi
#le = LabelEncoder()
#y = le.fit_transform(df[label_col])
#print(list(le.classes_))
#print(y)

# 7. Rimuovi colonne categoriche se presenti
X = X.select_dtypes(include=["float64", "int64"])

# 8. Standardizzazione 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Ridimensionamento per la CNN
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# 10. One-hot encoding binario
y_cat = to_categorical(y, num_classes=2)

# 11. Split dal dataset originale del Train dataset di Test
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, 
    y_cat, 
    test_size=0.2, 
    stratify=np.argmax(y_cat, axis=1), 
    random_state=42 )
#print("Distribuzione classi nel test set:", np.unique(np.argmax(y_test, axis=1), return_counts=True))

# 12. Bilanciamento del Train dataset mediante metodo ziopask undersample
y_train_int = np.argmax(y_train, axis=1)  
X_train, y_train = sm.balance_classes_undersample(X_train, y_train)

# 13. salvataggio dello scaler dal metodo dump 
save_scaler(scaler, "scaler_mono.pkl")

# 14. Definizione del modello di Machine Learning
input_shape = (None, X_train.shape[1], 1)
modelInit = cnnMonokernel(2, input_shape)
model = modelInit.build_cnn_mono()

sm.overwriteModelMap(model, "model_three_dim_kernel.txt")
plot_model(model, to_file="cnn_mono_map.png", show_shapes=True, show_layer_names=True)

# 15. callback per ogni ciclo di addestramento e quando smette di apprendere salviamo il migliore
callbacks_var = [
    EarlyStopping(patience=3, restore_best_weights=True), 
    ReduceLROnPlateau(patience=3, factor=0.5)             
]

# 16. Machine Learning
history = model.fit(
    X_train, 
    y_train,
    epochs = 20,           
    batch_size = 64,
    validation_data = (X_test, y_test),
#    class_weight = class_weights,
    callbacks = callbacks_var
)

# 17. Report di Valutazione ========
loss, acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
report = classification_report(y_true_labels, y_pred_labels)
sm.appendToTxt("model_three_dim_kernel.txt", report)
plot = sm.plotTrainingAccuracy(history, "Training Accuracy - CNN Three Dimentional Kernel")
plot.show()

# 18. Salva il modello con il nuovo formato keras
model.save("unclepaskm_cnn_mono.keras")