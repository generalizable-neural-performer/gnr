
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding:
    """
    GNR uses positional encoding in NeRF for coordinate embedding 
    """
    def __init__(self, d, num_freqs=10, min_freq=None, max_freq=None, freq_type='linear'):
        self.num_freqs = num_freqs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_type = freq_type
        self.create_embedding_fn(d)
        
    def create_embedding_fn(self, d):
        embed_fns = []
        out_dim = 0
        embed_fns.append(lambda x : x)
        out_dim += d
            
        N_freqs = self.num_freqs
        
        if self.freq_type == 'linear':
            min_freq = 0 if self.min_freq is None else self.min_freq
            max_freq = 2 ** (self.num_freqs-1) if self.max_freq is None else self.max_freq
            freq_bands = torch.linspace(min_freq*math.pi*2, max_freq*math.pi*2, steps=N_freqs) # linear freq band, Fourier expansion
        else:
            min_freq = 0 if self.min_freq is None else math.log2(self.min_freq)
            max_freq = self.num_freqs-1 if self.max_freq is None else math.log2(self.max_freq)
            freq_bands = 2.**torch.linspace(min_freq*math.pi*2, max_freq*math.pi*2, steps=N_freqs)    # log expansion

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class SphericalHarmonics:
    """
    GNR uses Sepherical Harmonics for view direction embedding 
    """
    def __init__(self, d = 3, rank = 3):
        assert d % 3 == 0
        self.rank = max([int(rank),0])
        self.out_dim= self.rank*self.rank * (d // 3)

    def Lengdre_polynormial(self, x, omx = None):
        if omx is None: omx = 1 - x * x
        Fml = [[]] *((self.rank+1)*self.rank//2)
        Fml[0] = torch.ones_like(x)
        for l in range(1, self.rank):
            b = (l * l + l) // 2
            Fml[b+l]  =-Fml[b-1]*(2*l-1)
            Fml[b+l-1]= Fml[b-1]*(2*l-1)*x
            for m in range(l,1,-1):
                Fml[b+m-2] = -(omx * Fml[b+m] + \
                         2*(m-1)*x * Fml[b+m-1]) / ((l-m+2)*(l+m-1))
        return Fml

    def SH(self, xyz):
        cs  = xyz[...,0:1]
        sn  = xyz[...,1:2]
        Fml = self.Lengdre_polynormial(xyz[...,2:3], cs*cs + sn*sn)
        H = [[]] *(self.rank*self.rank)
        for l in range(self.rank):
            b = l * l + l
            attr = np.sqrt((2*l+1)/math.pi/4)
            H[b] = attr * Fml[b//2]
            attr = attr * np.sqrt(2)
            snM = sn; csM = cs
            for m in range(1, l+1):
                attr = -attr / np.sqrt((l+m)*(l+1-m))
                H[b-m] = attr * Fml[b//2+m] * snM
                H[b+m] = attr * Fml[b//2-m] * csM
                snM, csM = snM*cs+csM*sn, csM*cs-snM*sn
        if len(H) > 0:
            return torch.cat(H, -1)
        else:
            return torch.Tensor([])

    def embed(self, inputs):
        return self.SH(inputs)
