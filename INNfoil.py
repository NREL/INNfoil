import os
import h5py
import numpy as np
import tensorflow as tf
from time import time
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')

class INNfoil(tf.keras.Model):
    def __init__(self, x, y, c, f, z, l, n_layers=15, permute_layers=None, input_shape=None, 
                                         scale_factors=None, model_path=None):
        '''
            Invertible neural network model and tools for forward/inverse evaluations and training 

            Parameters:
                x, y, c, f, z, l - TensorFlow input objects for various input/output features
                n_layers         - Number of invertible block layers
                permute_layers   - Numpy integer array that defines each layers node connections
                input_shape      - TensorShape object used to initialize the network weights
                W                - Numpy array defining the subspace-based dimension reduction of x
                scale_factors    - Python dictionary containing normalization information of data
                model_path       - Path to the saved model weights (in *.h5 format) to be loaded
        '''
        super(INNfoil, self).__init__()

        
        self.x, self.y, self.c, self.f, self.z, self.l = x, y, c, f, z, l
        self.xM, self.yM = x.shape[1], y.shape[1]
        self.cM, self.fM = c.shape[1], f.shape[1]
        self.zM = z.shape[1]

        self.scale_factors = scale_factors
        
        self.n_layers = n_layers
        
        # Define forward/inverse network permutations of nodes for each layer
        if permute_layers is None:
            self.permute_layers = tf.stack([tf.range(self.xM+self.yM) for i in range(n_layers)]) 
        self.inv_permute_layers = self.invert_perm(self.permute_layers)

        # Define forward functions that comprise each invertible block
        self.s0 = [self.dense_layer((self.xM+self.yM)/2, name='s0_{0:02d}'.format(i)) for i in range(n_layers)]
        self.t0 = [self.dense_layer((self.xM+self.yM)/2, name='t0_{0:02d}'.format(i)) for i in range(n_layers)]
        self.s1 = [self.dense_layer((self.xM+self.yM)/2, name='s1_{0:02d}'.format(i)) for i in range(n_layers)]
        self.t1 = [self.dense_layer((self.xM+self.yM)/2, name='t1_{0:02d}'.format(i)) for i in range(n_layers)]

        # Initialize network
        if input_shape is not None:
            self.build_model(input_shape)

        # Load previously trained network weights, if provided
        if model_path is not None:
            self.network_load_weights(model_path)

        self.optimizer = tf.keras.optimizers.Adam()

    def dense_layer(self, dim, name, expand_dim=3):
        # Template for forward functions containing 2 fully-connected layers
        initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
        regularizer = tf.keras.regularizers.l2(0.1)

        layer = tf.keras.Sequential(name=name)


        layer.add(tf.keras.layers.Dense(expand_dim*dim,
                                        use_bias=True,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer,
                                        name=name+'/dense0'))
        layer.add(tf.keras.layers.LeakyReLU(alpha=0.2, name=name+'/relu0'))
        layer.add(tf.keras.layers.Dense(dim,
                                        use_bias=True,
                                        kernel_initializer=initializer,
                                        bias_initializer=initializer,
                                        name=name+'/dense1'))
        layer.add(tf.keras.layers.LeakyReLU(alpha=0.2, name=name+'/reul1'))

        return layer

    def invert_perm(self, perm_array):
        # Given a forward permutation matrix, generate inverse permutations
        inv_perm_array = []
        for perm in tf.unstack(perm_array):
            inv_perm = np.zeros_like(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            inv_perm_array.append(tf.convert_to_tensor(inv_perm))
        return tf.stack(inv_perm_array)

    def build_model(self, input_shape):
        # Run the forward model with dummy data to initialize the network
        xx = tf.random.normal([1, input_shape[0]], dtype=tf.dtypes.float64)
        _ = self.forward_pass(xx)

    def forward_pass(self, x):
        # Connect individual pieces of invertible blocks in forward direction
        assert x.shape[1] % 2 == 0

        N, C = tf.shape(x)
        C2 = int(C/2)
        for i in range(self.n_layers):
            x = tf.gather(x, self.permute_layers[i, :], axis=1)
            x0, x1 = x[..., :C2], x[..., C2:]

            x0 = x0*tf.exp(self.s1[i](x1)) + self.t1[i](x1)
            x1 = x1*tf.exp(self.s0[i](x0)) + self.t0[i](x0)

            x = tf.concat([x0, x1], axis=-1)
        return x

    def inverse_pass(self, x):
        # Connect individual pieces of invertible blocks in inverse direction
        assert x.shape[1] % 2 == 0

        N, C = tf.shape(x)
        C2 = int(C/2)
        for i in range(self.n_layers):
            ii = self.n_layers - (i+1)
            x0, x1 = x[..., :C2], x[..., C2:]

            x1 = (x1 - self.t0[ii](x0))*tf.exp(-self.s0[ii](x0))
            x0 = (x0 - self.t1[ii](x1))*tf.exp(-self.s1[ii](x1))

            x = tf.concat([x0, x1], axis=-1)
            x = tf.gather(x, self.inv_permute_layers[ii, :], axis=1)
        return x

    def eval_forward(self, x, y):
        # Run forward model and partition output data
        out = self.forward_pass(tf.concat([x, y], axis=1))

        y_out,   out = out[..., :self.yM], out[..., self.yM:]
        c_out,   out = out[..., :self.cM], out[..., self.cM:]
        f_out, z_out = out[..., :self.fM], out[..., self.fM:]

        return y_out, c_out, f_out, z_out

    def eval_inverse(self, y, c, f, z):
        # Run inverse model and partition input data
        out = self.inverse_pass(tf.concat([y, c, f, z], axis=1))

        x_out, y_out = out[..., :self.xM], out[..., self.xM:]

        return x_out, y_out

    def network_save_weights(self, filename):
        # Customized function to save model weights and permutations
        perm_path = '/'.join(filename.split('/')[:-1])
        np.save('/'.join([perm_path, 'network_permutation.npy']), self.permute_layers)

        # Identifies all trainable weights in network
        building_blocks = [self.s0, self.t0, self.s1, self.t1]
        all_weights = [weight for block in building_blocks \
                              for layer in block \
                              for weight in layer.weights]

        # Writes each weight to an *.h5 file
        f = h5py.File(filename, 'w')
        for weight in all_weights:
            w_name = weight.name.split('/')
            w_path, w_name = '/'.join(w_name[:-1]), w_name[-1]
            try:
                f.create_group(w_path)
            except:
                pass
            f[w_path].create_dataset(w_name, data=weight.numpy())
        f.close()

    def network_load_weights(self, filename):
        # Customized function to load model weights and permutations
        perm_path = '/'.join(filename.split('/')[:-1])
        self.permute_layers = tf.convert_to_tensor(np.load('/'.join([perm_path, 'network_permutation.npy'])))
        self.inv_permute_layers = self.invert_perm(self.permute_layers)

        # Identifies all trainable weights in network
        building_blocks = [self.s0, self.t0, self.s1, self.t1]
        all_weights = [weight for block in building_blocks \
                              for layer in block \
                              for weight in layer.weights]

        # Assigns each weight tensor with the appropriate values
        f = h5py.File(filename, 'r')
        for weight in all_weights:
            weight.assign(f[weight.name][()])
        f.close()

    def k(self, x0, x1, h=1, ignore_diag=True):
        # Kernel function for MMD loss function
        N0, N1 = tf.shape(x0)[0], tf.shape(x1)[0]
        c0, c1 = x0.get_shape()[1], x1.get_shape()[1]
        assert c0 == c1

        x0, x1 = tf.reshape(x0, shape=(-1, 1, c0)), tf.reshape(x1, shape=(1, -1, c1))

        d = 1./(1. + tf.reduce_sum(((x0 - x1)/h)**2, axis=-1))
        if ignore_diag:
            d -= tf.eye(N0, N1, dtype=tf.float64)

        return d

    def MMD_z(self, z0, k_func):
        # Compute MMD loss on latent space variables compared to standard Gaussian
        N0, c = tf.shape(z0)[0], z0.get_shape()[1]
        N1 = N0
        z1 = tf.random.normal(shape=[N1, c], dtype=tf.float64)
        
        k00 = k_func(z0, z0)
        k01 = k_func(z0, z1)
        k11 = k_func(z1, z1)
        
        mmd_loss = tf.reduce_sum(k00)/tf.cast(N0*(N0-1), dtype=tf.float64) \
                 - 2*tf.reduce_sum(k01)/tf.cast(N1*N0, dtype=tf.float64) \
                    + tf.reduce_sum(k11)/tf.cast(N1*(N1-1), dtype=tf.float64)

        if tf.math.is_nan(mmd_loss):
            mu, sigma = tf.nn.moments(z0, axes=0)

            return tf.reduce_sum(mu**2) + tf.reduce_sum((sigma - 1.)**2)
        else:
            return mmd_loss

    def MMD_x(self, x0, x1, k_func):
        # Compute MMD loss on input space variables
        N0, c = tf.shape(x0)[0], x0.get_shape()[1]
        N1 = tf.shape(x1)[0]
        
        k00 = k_func(x0, x0)
        k01 = k_func(x0, x1)
        k11 = k_func(x1, x1)
        
        mmd_loss = tf.reduce_sum(k00)/tf.cast(N0*(N0-1), dtype=tf.float64) \
                 - 2*tf.reduce_sum(k01)/tf.cast(N1*N0, dtype=tf.float64) \
                    + tf.reduce_sum(k11)/tf.cast(N1*(N1-1), dtype=tf.float64)

        if tf.math.is_nan(mmd_loss):
            mu0, sigma0 = tf.nn.moments(x0, axes=0)
            mu1, sigma1 = tf.nn.moments(x1, axes=0)

            return tf.reduce_sum((mu0 - mu1)**2) + tf.reduce_sum((sigma0 - sigma1)**2)
        else:
            return mmd_loss

    def forward_supervised_loss(self, x, y, c, f, l):
        # Compute supervised losses of forward model
        y_out, c_out, f_out, z_out = self.eval_forward(x, y)
        y_loss = tf.reduce_mean((y_out - y)**2, axis=1)
        c_loss = tf.reduce_mean((c_out - c)**2, axis=1)

        # For f losses, we include implied coefficient of lift in the loss function
        f_out_norm = unnorm_data(f_out, self.scale_factors['f'])
        cd, cld = tf.reshape(tf.math.pow(10, f_out_norm[:, 0]), [-1, 1]), tf.reshape(f_out_norm[:, 1], [-1, 1])
        cl = norm_data(cd*cld, self.scale_factors['l'])
        f_loss = tf.reduce_mean((f_out - f)**2, axis=1) + 0.1*(cl - l)**2

        return y_loss, c_loss, f_loss

    def forward_supervised_train_step(self, x, y, c, f, l):
        # Perform forward supervised training step
        with tf.GradientTape() as tape:
            y_loss, c_loss, f_loss = self.forward_supervised_loss(x, y, c, f, l)
            y_loss, c_loss, f_loss = tf.reduce_mean(y_loss), tf.reduce_mean(c_loss), tf.reduce_mean(f_loss)

            grad = tape.gradient((y_loss, c_loss, f_loss), self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
            
    def forward_unsupervised_train_step(self, x, y):
        # Perform forward unsupervised training step
        with tf.GradientTape() as tape:
            y_out, c_out, f_out, z_out = self.eval_forward(x, y)
            z_loss = 1000.*self.MMD_z(z_out, self.k)

            if not tf.math.is_nan(z_loss):
                grad = tape.gradient(z_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def inverse_supervised_train_step(self, x, y, c, f):
        # Perform inverse supervised training step
        with tf.GradientTape() as tape:
            z = tf.random.normal([tf.shape(x)[0], self.zM], dtype=tf.float64)
            x_out, y_out = self.eval_inverse(y, c, f, z)
            
            y_loss = tf.reduce_mean((y_out - y)**2)

            grad = tape.gradient(y_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def inverse_unsupervised_train_step(self, x, y, c, f):
        # Perform inverse unsupervised training step
        with tf.GradientTape() as tape:
            z = tf.random.normal([tf.shape(x)[0], self.zM], dtype=tf.float64)
            x_out, y_out = self.eval_inverse(y, c, f, z)
            x_loss = 1000.*self.MMD_x(x_out, x, self.k)

            x_loss += tf.reduce_mean((abs(x_out[tf.abs(x_out) > 1.]) - 1.)**2)

            if not tf.math.is_nan(x_loss):
                grad = tape.gradient(x_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    def compute_dataset_loss(self, ds, N_reps=1):
        yf_loss, c_loss, f_loss, z_loss = 0., 0., 0., 0.
        x_loss, yi_loss = 0., 0.
        N_loss = 0.
        for xx, yy, cc, ff, ll in ds:
            N_batch = xx.shape[0]

            yf_l, c_l, f_l = self.forward_supervised_loss(xx, yy, cc, ff, ll)
            _, _, _, z_out = self.eval_forward(xx, yy)

            yf_l, c_l, f_l = tf.reduce_sum(yf_l), tf.reduce_sum(c_l), tf.reduce_sum(f_l)
            z_l = self.MMD_z(z_out, self.k).numpy()

            xx_reps, yy_reps = tf.repeat(xx, N_reps, axis=0), tf.repeat(yy, N_reps, axis=0)
            cc_reps, ff_reps = tf.repeat(cc, N_reps, axis=0), tf.repeat(ff, N_reps, axis=0)
            zz_reps = tf.random.normal([tf.shape(xx_reps)[0], self.zM], dtype=tf.float64)
            x_out, y_out = self.eval_inverse(yy_reps, cc_reps, ff_reps, zz_reps)
            x_l = self.MMD_x(x_out, xx_reps, self.k).numpy()
            yi_l = np.sum((y_out - yy_reps)**2)

            yf_loss += yf_l
            c_loss += c_l
            f_loss += f_l
            z_loss += (N_batch*z_l)
            x_loss += (N_batch*N_reps*x_l)
            yi_loss += yi_l
            N_loss += N_batch

        yf_loss, c_loss = yf_loss/N_loss, c_loss/N_loss
        f_loss, z_loss  = f_loss/N_loss,  z_loss/N_loss
        x_loss, yi_loss = x_loss/(N_loss*N_reps), yi_loss/(N_loss*N_reps)

        return yf_loss, c_loss, f_loss, z_loss, x_loss, yi_loss
            
    def fit(self, train_ds, test_ds, learning_rate=1e-4, decay_rate=0.999, epochs=100, N_reps=1, 
                  print_every=10, save_every=100, save_model_path=None):
        # Training function for network
        self.optimizer.learning_rate = learning_rate

        iters = 0
        for epoch in range(1, epochs+1):
            start_time = time()

            print('Beginning epoch', epoch, '...')

            # Initialilze epoch losses 
            yf_loss, c_loss, f_loss, z_loss = 0., 0., 0., 0.
            x_loss, yi_loss = 0., 0.
            N_loss = 0.
            for xx, yy, cc, ff, ll in train_ds:
                N_batch = xx.shape[0]

                # Repeat data to improve inverse losses (with respect to multiple realizations of z)
                xx_reps, yy_reps = tf.repeat(xx, N_reps, axis=0), tf.repeat(yy, N_reps, axis=0)
                cc_reps, ff_reps = tf.repeat(cc, N_reps, axis=0), tf.repeat(ff, N_reps, axis=0)

                # Run forward training steps and save forward losses
                self.forward_supervised_train_step(xx, yy, cc, ff, ll)
                self.forward_unsupervised_train_step(xx, yy)

                # Run inverse training steps and save inverse losses
                self.inverse_supervised_train_step(xx_reps, yy_reps, cc_reps, ff_reps)
                self.inverse_unsupervised_train_step(xx_reps, yy_reps, cc_reps, ff_reps)
                
                # Every print_every iterations, print this steps losses
                if iters % print_every == 0:
                    yf_iter_loss, c_iter_loss, f_iter_loss = self.forward_supervised_loss(xx, yy, cc, ff, ll)
                    yf_iter_loss = tf.reduce_sum(yf_iter_loss)
                    c_iter_loss = tf.reduce_sum(c_iter_loss)
                    f_iter_loss = tf.reduce_sum(f_iter_loss)
                    y_out, c_out, f_out, z_out = self.eval_forward(xx, yy)
                    z_iter_loss = self.MMD_z(z_out, self.k).numpy()

                    zz_reps = tf.random.normal([tf.shape(xx_reps)[0], self.zM], dtype=tf.float64)
                    x_out, y_out = self.eval_inverse(yy_reps, cc_reps, ff_reps, zz_reps)
                    x_iter_loss = self.MMD_x(x_out, xx_reps, self.k).numpy()
                    yi_iter_loss = np.sum((y_out - yy_reps)**2)

                    print('Iter:', iters)
                    print('Forward Losses:')
                    print('L_y = {0:04f}, '.format(yf_iter_loss/N_batch), end=' ')
                    print('L_c = {0:04f}, '.format(c_iter_loss/N_batch), end=' ')
                    print('L_f = {0:04f}, '.format(f_iter_loss/N_batch), end=' ')
                    print('L_z = {0:04f}'.format(z_iter_loss))
                    print('Inverse Losses:')
                    print('L_x = {0:04f}, '.format(x_iter_loss), end=' ')
                    print('L_y = {0:04f}\n'.format(yi_iter_loss/(N_batch*N_reps)))

                    # If losses start producing NaNs, break the training procedure
                    if tf.math.is_nan(yf_iter_loss) \
                       or tf.math.is_nan(c_iter_loss) \
                       or tf.math.is_nan(f_iter_loss) \
                       or tf.math.is_nan(z_iter_loss) \
                       or tf.math.is_nan(x_iter_loss) \
                       or tf.math.is_nan(yi_iter_loss):
                        assert False, 'Loss function has produced NaNs'

                iters += 1

            yf_loss, c_loss, f_loss, z_loss, x_loss, yi_loss = self.compute_dataset_loss(train_ds, N_reps=N_reps)

            # Print out epoch training losses
            print('-- Training --')
            print('Forward Losses:')
            print('L_y = {0:04f}, '.format(yf_loss), end=' ')
            print('L_c = {0:04f}, '.format(c_loss), end=' ')
            print('L_f = {0:04f}, '.format(f_loss), end=' ')
            print('L_z = {0:04f}'.format(z_loss))
            print('Inverse Losses:')
            print('L_x = {0:04f}, '.format(x_loss), end=' ')
            print('L_y = {0:04f}'.format(yi_loss))
            
            if save_model_path is not None and ((epoch % save_every) == 0):
                model_epoch = save_model_path+'/{0:05d}'.format(epoch)
                if not os.path.exists(model_epoch):
                    os.makedirs(model_epoch)
                self.network_save_weights('/'.join([model_epoch, 'inn.h5']))

            self.optimizer.learning_rate = decay_rate*self.optimizer.learning_rate

            if test_ds is not None:
                # Loop through test data and compute losses (without performing training steps)
                yf_loss, c_loss, f_loss, z_loss, x_loss, yi_loss = self.compute_dataset_loss(test_ds, N_reps=N_reps)

                # Print out epoch testing losses
                print('-- Testing --')
                print('Forward Losses:')
                print('L_y = {0:04f}, '.format(yf_loss), end=' ')
                print('L_c = {0:04f}, '.format(c_loss), end=' ')
                print('L_f = {0:04f}, '.format(f_loss), end=' ')
                print('L_z = {0:04f}'.format(z_loss))
                print('Inverse Losses:')
                print('L_x = {0:04f}, '.format(x_loss), end=' ')
                print('L_y = {0:04f}'.format(yi_loss))

            print('Epochs {}'.format(epoch)+' time to complete: {0:02f}\n'.format(time() - start_time), flush=True)

    def inverse_design(self, cd, clcd, stall_margin, thickness, Re, 
                             N=10, process_samples=True):
        # Normalize and format conditional inputs for inverse model evaluation
        y_val, c_val = np.array([Re]), np.array([thickness])
        f_val = np.array([np.log10(cd), clcd, stall_margin] + [0. for _ in range(self.fM-3)])

        y_val = norm_data(y_val.reshape((-1, self.yM)), self.scale_factors['y'])
        c_val = norm_data(c_val.reshape((-1, self.cM)), self.scale_factors['c'])
        f_val = norm_data(f_val.reshape((-1, self.fM)), self.scale_factors['f'])

        # If process_sampes, then we must oversample the inverse direction to find best shapes
        NN = 10*N if process_samples else N

        # Run inverse model with randomly sampled latent variables (z)
        y_val = tf.repeat(y_val, NN, axis=0)
        c_val = tf.repeat(c_val, NN, axis=0)
        f_val = tf.repeat(f_val, NN, axis=0)
        z_val = tf.random.normal([NN, self.zM], dtype=tf.float64)
        x_inv, _ = self.eval_inverse(y_val, c_val, f_val, z_val)

        # Sort generated shapes by errors in network forward prediction
        if process_samples:
            y_out, c_out, f_out, _ = self.eval_forward(x_inv, y_val)

            y_val = unnorm_data(y_val, scale_factor=self.scale_factors['y'])
            c_val = unnorm_data(c_val, scale_factor=self.scale_factors['c'])
            f_val = unnorm_data(f_val, scale_factor=self.scale_factors['f'])
            y_val, c_val, f_val = y_val.numpy(), c_val.numpy(), f_val.numpy()
            
            y_out = unnorm_data(y_out, scale_factor=self.scale_factors['y'])
            c_out = unnorm_data(c_out, scale_factor=self.scale_factors['c'])
            f_out = unnorm_data(f_out, scale_factor=self.scale_factors['f'])
            y_out, c_out, f_out = y_out.numpy(), c_out.numpy(), f_out.numpy()

            f_val[:, 0], f_out[:, 0] = np.power(10., f_val[:, 0]), np.power(10., f_out[:, 0])

            # Compute relative errors for each output
            y_err = np.sum(np.abs((y_val - y_out)/y_val), axis=1)
            c_err = np.sum(np.abs((c_val - c_out)/c_val), axis=1)
            f_err = np.sum(np.abs((f_val - f_out)/f_val), axis=1)

            x_inv = tf.convert_to_tensor(x_inv.numpy()[np.argsort(y_err+c_err+f_err)[:N], :])

        # Maps normalilzed CST parameters back to physical space
        x_inv = unnorm_data(x_inv, scale_factor=self.scale_factors['x'])
        cst, alpha = x_inv[:, :-1].numpy(), x_inv[:, -1:].numpy()

        # Appends the trailing edge thickness
        if 'trailing_edge' in self.scale_factors:
            trailing_edge = np.array(self.scale_factors['trailing_edge'][:2])
            trailing_edge = tf.repeat(trailing_edge.reshape((1, -2)), N, axis=0)
            cst = np.concatenate((cst, trailing_edge), axis=1)

        return cst, alpha


