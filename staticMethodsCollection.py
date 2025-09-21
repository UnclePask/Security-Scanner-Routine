'''
Created on 13 set 2025

@author: pasquale
'''

from glob import glob 
from sklearn.utils import resample
from keras_flops import get_flops
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Conv2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

class filesMethods(object):
    '''
    classdocs
    '''
    
    @staticmethod
    def overwriteModelMap(model: any, titleFileTxt: str):
        try:
            with open(titleFileTxt, "w") as f:
                model.summary(print_fn = lambda line: f.write(line + "\n")) #che follia :D
        except Exception:
            traceback.print_exc()
            
    @staticmethod
    def appendToTxt(titleFileTxt: str, data_string: str):
        try:
            with open(titleFileTxt, "a") as f:
                f.write(data_string)
        except Exception:
            traceback.print_exc()
    
    @staticmethod
    def appendDictToTxt(titleFileTxt: str, data_dict: any):
        try:
            with open(titleFileTxt, "a") as f:
                f.write(str(data_dict) + "\n")
                f.write("\n")
        except Exception:
            traceback.print_exc()
    
    @staticmethod
    def balance_classes_undersample(x_train, y_train):
        """
        Eseguo undersampling sulla classe BENIGN (0) per bilanciare il dataset.
        Non uso class_weight di sklearn
        Funziona con dati CNN (X a 3 dimensioni) e y one-hot encoded (es. [1, 0] / [0, 1]).
        """
        # Appiattisci X per concatenare con y
        x_flat = x_train.reshape(x_train.shape[0], -1)

        # Unisci X e y in un solo array
        data = np.concatenate((x_flat, y_train), axis=1)

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
        x_bal = balanced_data[:, :-2].reshape(-1, x_train.shape[1], 1)
        y_bal = balanced_data[:, -2:]

        return x_bal, y_bal
    
    @staticmethod
    def readDatasetParquet(path: str):
        '''
        Percorso ai file Parquet (modifica il path a seconda dei tuoi file)
        '''
        file_paths = glob(path)
        dfs = [pd.read_parquet(f) for f in file_paths]
        df = pd.concat(dfs, ignore_index=True)
#        print(df.columns)
        df = df.dropna() 
        return df
    
    @staticmethod
    def plotTrainingAccuracy(modelFit: any, titlePlot: str):
        '''
        plot del modello dopo training
        $args$ modelFit è una matrice numpy
        $args$ titlePlot è una stringa titolo del grafico
        '''
        plt.plot(modelFit.history["accuracy"], label="Train")
        plt.plot(modelFit.history["val_accuracy"], label="Validation")
        plt.title(titlePlot)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        try:
            name_file = titlePlot + ".png"
            plt.savefig(name_file)
        except Exception as e:
            print("\nError (3): " + e)
        
        return plt
    
    @staticmethod
    def plotRoughSet(pos: pd.DataFrame(), neg: pd.DataFrame(), bnd: pd.DataFrame(), model_name: str):
        labels = ["Positivi", "Negativi", "Boundary"]
        values = [pos.index.size, neg.index.size, bnd.index.size]
        colors = ["green", "red", "orange"]
        plt.bar(labels, values, color = colors)
        plt.ylabel("Numero pacchetti individuati")
        plt.xlabel("Rough Sets")
        titlePlot = "Confronto numero di pacchetti per Rough Set " + model_name
        plt.title(titlePlot)
        
        try:
            name_file = titlePlot + ".png"
            plt.savefig(name_file)
        except Exception as e:
            print("\nError (3): " + e)
        
        return plt
    
class modelAnalysis(object):
    '''
    metodi statici per analisi della complessità
    
        h = profondità
        w = lunghezza
        c = numero di canali
        in = input
        out = output
    ''' 
    @staticmethod
    def getMacsConvolution(layer: any):
        '''
        Calcola i MACs per secondo la formula: 
        dim out x canali x dim kernel
        un layer dato l'input_shape 
        '''
        # prendo gli shape dentro il layer
        input_shape = layer.input.shape
        output_shape = layer.output.shape
        
        #prendo il numero di canali nello shape input
        c_in = input_shape[-1]
        
        #prendo profondità e altezza e canali dello shape in output (..., h, w, c)
        h_out, w_out, c_out = output_shape[1:]
        #verifico h_out = None (nel caso del modello non compilato) 
        # con sintassi if "fantastica" :)
        h_out = 1 if h_out is None else h_out 
        
        #leggo la tupla con le misure del kernel del layer
        kernel_dims = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)
        k_h, k_w = kernel_dims
        
        #calcolo i macs secondo formula per prodotto convoluzionale
        #Fonte: Towards Image Recognition with Extremely Low FLOPs (Yunsheng Li et al.)
        macs = h_out * w_out * c_out * (k_h * k_w * c_in)
        
        #tutte le funzioni di calcolo cuncluse e testate: 00:16 - 20.09.2025
        return macs
    
    @staticmethod
    def getMacsDense(layer: any):
        '''
        Calcola i MACs per lo strato fully connected  
        n*n
        '''
        # la dimensione del batch per gli strati dense non da contributo
        dim_batch, n_in = layer.kernel.shape.dims
        dim_batch = 1
        
        n_out = layer.units
        macs = n_in * n_out
        
        #return come dimensione va in dump 
        #se fa operazioni algebriche con float
        return macs
    
    @staticmethod
    def getMacsPoolingLayers(layer: any):
        '''
        Calcola i MACs per secondo la formula: 
        dim out x canali x dim kernel
        un layer dato l'input_shape 
        '''
        # bisogna distinguere le istanze di pooling
        if isinstance(layer, GlobalAveragePooling2D):
        # recupero solo h e w perché per global pooling i canali sono "implicit"
            h_out, w_out = layer.output.shape
            c_out = getattr(layer, 'filters', 1)
        elif isinstance(layer, GlobalMaxPooling2D):
            h_out, w_out = layer.output.shape
            c_out = getattr(layer, 'filters', 1)
        elif isinstance(layer, MaxPooling2D):
            _, h_out, w_out, c_out = layer.output.shape
        # con sintassi if "fantastica" :)
        h_out = 1 if h_out is None else h_out
        
        # altrimenti uso c_out = 1 perchè i canali in uscita sono impliciti e trascurabili
        k_h = 1
        k_w = 1  # default se non esiste pooling esempio per i metodi glòobal
        if hasattr(layer, "pool_size"):
            k_h, k_w = layer.pool_size

        macs = h_out * w_out * c_out * (k_h * k_w)

        return macs
    
    @staticmethod
    def getMacsMultiplyAdd(layer: any):
        '''
        Calcola macs dai flops
        '''
        # prendiamo shape in output
        # prendi lo shape dell'output
        output_shape = layer.output.shape

        # sostituisco None con 1 perché non devono dare contributo nel prodotto
        ten_shape = [1 if i is None else i for i in output_shape]
        
        # faccio il prodotto di tutte le entrate del tensore
        flops = int(np.prod(ten_shape))
        
        #perché macs flops in realtà quando esce dal calcolo fully connected 
        #è una dimensione non un semplice int
        #quindi poi flops = flops + (macs * 2) va in dump
        macs = int(flops / 2) 
        
        return macs
    
    @staticmethod
    def getFlops(model: Model):
        '''
        calcola le operazioni in virgola mobile per singolo input 
        con la libreria standard get_flops
        
        non va sto maledetto senza downgrade
        '''
        flops = get_flops(model, batch_size = 1)
        return flops
    
    
    
    @staticmethod
    def spatialAnalysis(input_shape: tuple, model: Model) -> tuple[int, float, dict[any, int]]:
        # trovo le attivazioni di tutti i layer
        out = [layer.output for layer in model.layers]
        act = Model(inputs=model.input, outputs=out)
        
        # genero un vettore di test con dati casuali 
        test = np.random.randn(1, *input_shape).astype(np.float32)
        
        #prendo gli output di tutti i layer convoluzionali
        conv_layers = [layer for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]
        activation_model = Model(inputs=model.input, outputs=[layer.output for layer in conv_layers])
        
        # faccio il predict con i dati di test casuali
        act_tensor = activation_model.predict(test)
        
        # calcolo le attivazioni totali con le relative entrate
        total_act = 0
        lay_coll = {}
        i = 0
        for layer, single_act in zip(model.layers, act_tensor):
            shape = single_act.shape
            num_elem_act = np.prod(shape)
            total_act = total_act + num_elem_act
            i = i + 1
            lay_coll[layer.name] = {
                "shape": shape,
                "num_elem_act": num_elem_act
                }
        
        # stimo la memoria occupata in megabyte
        mb_value = total_act * 4 / 1024**2
        
        return total_act, mb_value, lay_coll