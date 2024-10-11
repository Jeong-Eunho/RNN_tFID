import numpy as np
import torch

def fftN(X,axis_in):
    return np.fft.fftshift(np.fft.fft(X,axis=axis_in),axes=axis_in)

def ifftN(X,axis_in):
    return np.fft.ifft(np.fft.ifftshift(X,axes=axis_in),axis=axis_in)

def fftT(X,axis_in):
    return torch.fft.fftshift(torch.fft.fft(X,dim=axis_in),dim=axis_in)

def ifftT(X,axis_in):
    return torch.fft.ifft(torch.fft.ifftshift(X,dim=axis_in),dim=axis_in)

def TtoN(X):
    return X.cpu().detach().numpy()

def Normalize(X_fid_obj,X_tfid):
    return (X_fid_obj/(np.max(np.abs(X_tfid[:,:,0]+1j*X_tfid[:,:,1]),axis=1)[:,np.newaxis,np.newaxis]))

def returnNormalize(X_fid_obj,X_tfid):
    return (X_fid_obj[:,:,0]+1j*X_fid_obj[:,:,1])*(np.max(np.abs(X_tfid[:,:,0]+1j*X_tfid[:,:,1]),1)[:, np.newaxis])
