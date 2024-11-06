import archVAE_D as av
import archMLP_D as am
import numpy as np
import readerGraphics_D as rg
from sklearn.preprocessing import MinMaxScaler
from cv2 import split
#%%
def norm_inputs():
    XT = np.load('airfoilsCSTplus.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(XT)
    return normXT, scaler 

def predictGraphs(Xtest):
    vae = av.ensembleVAE()
    vae.load_weights('vaes_7_1500.weights.h5')    
    mlp = am.MLP()
    mlp.load_weights('mlp_7A13_1500.weights.h5')
    
    _, scaler = norm_inputs()
    X_Test = scaler.fit_transform(Xtest)
    
    zPred = mlp.predict(X_Test)
    Ygraphs = vae.decoder.predict(zPred)
    return Ygraphs

def predictMaxE(Xtest):
    graphs = predictGraphs(Xtest)
    E = np.array([])
    alpha = np.array([])
    for i in range(len(graphs)):
        _, _, graphE = split(graphs[i])
        e, aoa = rg.searchMaxE(graphE)
        E = np.append(E, e)
        alpha = np.append(alpha, aoa)
    return E, alpha