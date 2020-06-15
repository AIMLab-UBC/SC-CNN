import spams
import cv2
import numpy as np
from matplotlib import pyplot as plt

def Beer_Lamber(image):

    temp_image = image.reshape(-1, image.shape[2])

    V = np.log(255)- np.log(temp_image+1)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #
    luminlayer = lab_image[:, :, 0]
    luminlayer = luminlayer.reshape(-1, 1)

    threshold = 0.9
    index = np.where(luminlayer / 255 < threshold)
    index = index[0]
    temp_new = temp_image[index, :]

    VformW = np.log(255)- np.log(temp_new+1)

    return V, VformW


def define_params(nstains, lambda1, batch):

    param = dict()

    param['mode'] = 2
    param['lambda1'] = lambda1
    param['posAlpha'] = True                    # positive stains
    param['posD'] = True                        # positive staining matrix
    param['modeD'] = 0                          # {W in Real^{m x n}  s.t.  for all j,  ||d_j||_2^2 <= 1 }
    param['verbose'] = False
    param['K'] = nstains                        # No. of stain = 2
    param['iter'] = 200                         # 20-50 is OK
    param['batchsize'] = batch

    param2 = dict()

    param2['lambda1'] = lambda1
    param2['pos'] = True
    param2['mode'] = 2

    return param, param2

def get_staincolor_sparsenmf(v, **param):

    # Params
    # Learning W through sparse NMF
    Ws = spams.trainDL(v.T, **param)

    # Arranging H stain color vector as first column and then the second column
    # vectors as E stain color vector
    Ws = Ws.T
    Ws = Ws[Ws[:,1].argsort()]

    return Ws.T

def estH(v, Ws, param, param2, nrows, ncols):

    Hs_vec = spams.lasso(v.T, Ws, **param2).T
    Hs_vec = Hs_vec.todense()

    Hs_vec = np.array(Hs_vec)
    Hs = Hs_vec.reshape(nrows, ncols, param['K'])
    iHs = []

    for i in range(param['K']):

        vdAS =  np.dot(Hs_vec[:, i].reshape(-1,1), Ws[:, i].reshape(-1,1).T)
        temp = 255 * np.exp(-1*vdAS).reshape(nrows, ncols, 3)
        temp = temp.astype('uint8')
        iHs.append(temp)

    Irecon = np.dot(Hs_vec, Ws.T)
    Irecon = 255 * np.exp(-1*Irecon).reshape(nrows, ncols, 3)
    Irecon = temp.astype('uint8')

    return Hs, iHs, Irecon

def prctile(x,p):

    col = x.shape[1]
    result = np.zeros((1, col))
    for i in range(col):
        result[0, i] = np.percentile(x[:, i], p, interpolation='midpoint')

    return result

def SCN(source, Hta, Wta, Hso):

    Hso = Hso.reshape(-1, Hso.shape[2])
    Hta = Hta.reshape(-1, Hta.shape[2])

    Hso_Rmax = prctile(Hso, 99)
    Hta_Rmax = prctile(Hta, 99)

    normfac = np.divide(Hta_Rmax, Hso_Rmax)
    Hsonorm = np.multiply(Hso, np.tile(normfac, (Hso.shape[0], 1)))

    Ihat = np.dot(Wta, Hsonorm.T)

    Ihat = (Ihat.T).reshape(source.shape[0], source.shape[1], 3)
    sourcenorm = 255 * np.exp(-1*Ihat)
    sourcenorm = sourcenorm.astype('uint8')

    return sourcenorm
