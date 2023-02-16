import numpy as np
from   sklearn.metrics.pairwise import rbf_kernel
from   sklearn.datasets import make_s_curve


def example1(gamma=1, n_samples=100):

    X, t = make_s_curve(n_samples, noise=0.1)
    X = X[t.argsort()]
    K = rbf_kernel(X, gamma=(1/(2*(10*gamma)**2))) 
    return K, X

def example2(gamma=1, n_samples=100):
    
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot
    
    dt = 0.01
    num_steps = n_samples
    
    # Need one more for the initial values
    xs = np.empty(num_steps )
    ys = np.empty(num_steps )
    zs = np.empty(num_steps )
    
    # Set initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps-1):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    
    X = np.vstack((xs,ys,zs)).T
    K = rbf_kernel(X, gamma=(1/(2*(10*gamma)**2))) 
    return K, X

def mnist(gamma=1, n_samples=100):
    from keras.datasets import mnist
    from sklearn.preprocessing import MinMaxScaler

    (X, y), (_ , _) = mnist.load_data()
    
    # use part of data
    X=X[0:n_samples,:,:].reshape(n_samples,-1)
    y=y[0:n_samples].flatten()
    
    X = MinMaxScaler().fit_transform(X)
    K = rbf_kernel(X, gamma=(1/(2*(10*gamma)**2))) 
    return K, X
    
    