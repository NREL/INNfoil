import h5py
import numpy as np

def norm_data(data, scale_factor=None):
    # Function to normalize the input data
    # Computes appropriate scale_factors from data if not provided
    if scale_factor[2] == 'standard':
        if scale_factor is None:
            mu, sig = np.mean(data, axis=0), np.std(data, axis=0)
            sig[sig == 0.] = 1.
            scale_factor = [mu, sig, 'standard']
        data = (data - scale_factor[0])/scale_factor[1]

    elif scale_factor[2] == 'minmax':
        if scale_factor is None:
            scale_factor = [np.min(data, axis=0), np.max(data, axis=0), 'minmax']
            tf = (scale_factor[0] == scale_factor[1])
            for i, tf_i in enumerate(tf):
                if tf_i:
                    if scale_factor[0][i] != 0.:
                        scale_factor[0][i], scale_factor[1][i] = 0., scale_factor[1][i]/2
                    else:
                        scale_factor[0][i], scale_factor[1][i] = -1., 1.
        data = 2.*(data - scale_factor[0])/(scale_factor[1] - scale_factor[0]) - 1.

    else:
        assert False, 'Scaling type invalid.'

    return data

def unnorm_data(data, scale_factor):
    # Function to unnormalize the input data to physical space
    if scale_factor[2] == 'standard':
        data = scale_factor[1]*data + scale_factor[0]

    elif scale_factor[2] == 'minmax':
        data = 0.5*(scale_factor[1] - scale_factor[0])*(data + 1.) + scale_factor[0]

    else:
        assert False, 'Scaling type invalid.'
    
    return data

def load_scale_factors():
    # Function to load previously saved scale_factors
    scale_factors = {}
    with h5py.File('model/scale_factors.h5', 'r') as f:
        for key in f:
            f_key = f[key]
            if f_key['type'][()] == 'minmax':
                scale_factors[key] = [f_key['min'][()], f_key['max'][()], 'minmax']
            else:
                scale_factors[key] = [f_key['mu'][()], f_key['sigma'][()], 'standard']

    return scale_factors


