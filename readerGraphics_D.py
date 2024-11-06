import numpy as np

def searchClwithAlpha(alpha, graph):    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 14
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    
    pxc = np.argmax(graph[:,px_alpha])
    cl1 = 1.6
    pxc1 = 0
    cl2 = 0.0
    pxc2 = 255
    cl = (pxc-pxc1)*(cl2-cl1)/(pxc2-pxc1)+cl1
    return cl

def searchCdwithAlpha(alpha, graph):    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 14
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    
    pxc = np.argmax(graph[:,px_alpha])
    cd1 = 0.04
    pxc1 = 0
    cd2 = 0.0
    pxc2 = 255
    cd = (pxc-pxc1)*(cd2-cd1)/(pxc2-pxc1)+cd1
    return cd

def searchEwithAlpha(alpha, graph):    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 14
    pxa2 = 255 
    px_alpha = (alpha-alpha1)*(pxa2-pxa1)/(alpha2-alpha1)+pxa1
    px_alpha = round(px_alpha)
    
    pxc = np.argmax(graph[:,px_alpha])
    E1 = 100
    pxc1 = 0
    E2 = 1
    pxc2 = 255
    E = (pxc-pxc1)*(E2-E1)/(pxc2-pxc1)+E1
    return E

def searchMaxE(graph):
    graphFil = np.where(graph>0.1, graph, 0)
    amax = np.argmax(graphFil, axis=0)
    amax = np.where(amax>0, amax, 255)
    pxc = min(amax)
    pxa = round(np.mean(np.where(amax==pxc)[0]))
    
    E1 = 100
    pxc1 = 0
    E2 = 1
    pxc2 = 255
    E = (pxc-pxc1)*(E2-E1)/(pxc2-pxc1)+E1
    
    alpha1 = -4
    pxa1 = 0
    alpha2 = 14
    pxa2 = 255
    alpha = (pxa-pxa1)*(alpha2-alpha1)/(pxa2-pxa1)+alpha1
    return E, alpha