import os
import numpy as np
import pandas as pd
import tensorflow as tf
from INNfoil import INNfoil
from time import strftime
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

load_model_path = '/'.join(['model', 'inn.h5'])
save_model_path = '/'.join(['model', 'trainingExample_'+strftime('%Y%m%d-%H%M%S')])
scale_factors = load_scale_factors()

# Load and preprocess data
df = pd.read_csv('data/test_data.csv', index_col=0)

cst = df[[str(i) for i in range(20)]].to_numpy()
cst_te = cst[:, -2:]
if np.all(np.min(cst_te, axis=0) == np.max(cst_te, axis=0)):
    cst = cst[:, :-2]
thickness = df['t/c max'].to_numpy().reshape((-1, 1))
Re = df['rey'].to_numpy().reshape((-1, 1))
alpha = df['aoa'].to_numpy().reshape((-1, 1))
stall = df['stall_margin'].to_numpy().reshape((-1, 1))
cl = df['cl'].to_numpy().reshape((-1, 1))
cd = df['cd'].to_numpy().reshape((-1, 1))
cld = df['l/d'].to_numpy().reshape((-1, 1))

x = norm_data(np.concatenate((cst, alpha), axis=1),
              scale_factor=scale_factors['x'])
y = norm_data(Re, scale_factor=scale_factors['y'])
c = norm_data(thickness, scale_factor=scale_factors['c'])
f = norm_data(np.concatenate((np.log10(cd), cld, stall), axis=1),
              scale_factor=scale_factors['f'])
l = norm_data(cl, scale_factor=scale_factors['l'])

# Set input/output data dimensions
xM = x.shape[1]
yM, cM, fM = y.shape[1], c.shape[1], f.shape[1]
zM = xM - (cM+fM)
lM = cl.shape[1]

x_in = tf.keras.Input(shape=(xM,))
y_in = tf.keras.Input(shape=(yM,))
c_in = tf.keras.Input(shape=(cM,))
f_in = tf.keras.Input(shape=(fM,))
z_in = tf.keras.Input(shape=(zM,))
l_in = tf.keras.Input(shape=(lM,))

# Initialize model
model = INNfoil(x_in, y_in, c_in, f_in, z_in, l_in,
                input_shape=tf.TensorShape([xM+yM]),
                scale_factors=scale_factors,
                model_path=load_model_path)

# Train model
ds = tf.data.Dataset.from_tensor_slices((x, y, c, f, l)).batch(50)
model.fit(ds, ds, 
          learning_rate=1e-4,
          decay_rate=0.99,
          N_reps=5,
          epochs=10,
          print_every=5,
          save_every=2,
          save_model_path=save_model_path)

# Run network inversion
cd = 0.015
clcd = 83.
stall_margin = 5.
thickness = 0.25
Re = 9000000.
cst, alpha = model.inverse_design(cd, clcd, stall_margin, thickness, Re, 
                                  N=100, process_samples=True)


