'''
Created on 18 set 2025

@author: pasquale
'''
from keras.models import load_model
from keras.layers import Conv2D, Add, GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply
from staticMethodsCollection import modelAnalysis as ma
from staticMethodsCollection import filesMethods as sm

input_shape = ()
model = load_model("unclepaskm_cnn_multi.keras", compile = False)
sm.overwriteModelMap(model, "model_ndim_analysis.txt")

#flops = ma.getFlops(model) - bug: _flops_add (libreria registory)
flops = 0
for layer in model.layers:
    macs = 0
    if isinstance(layer, Conv2D):
        macs = ma.getMacsConvolution(layer)
    elif isinstance(layer, Dense):
        macs = ma.getMacsDense(layer)
    elif isinstance(layer, (MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D)):
        macs = ma.getMacsPoolingLayers(layer)
    elif isinstance(layer, (Add, Multiply)):
        macs = ma.getMacsMultiplyAdd(layer)
    
    flops = flops + (macs * 2)
        
    if macs != 0:
        value_in_file = layer.name + " macs = " + str(macs) + "\n" 
        sm.appendToTxt("model_ndim_analysis.txt", value_in_file)
    else:
        value_in_file = layer.name + " negligible \n" 
        sm.appendToTxt("model_ndim_analysis.txt", value_in_file)

value_in_file = model.name + " FLOPs = " + str(flops) + "\n"
sm.appendToTxt("model_ndim_analysis.txt", value_in_file)

input_shape = (1, 16, 1)
total_act, mb_value, lay_coll = ma.spatialAnalysis(input_shape, model)

value_in_file = "Spatial Analysis: \nTotal Activations = " + str(total_act) + "\n" +  "Memory use (mb) = " + str(mb_value) + "\n"
print(value_in_file)
sm.appendToTxt("model_ndim_analysis.txt", value_in_file)
sm.appendDictToTxt("model_ndim_analysis.txt", lay_coll)