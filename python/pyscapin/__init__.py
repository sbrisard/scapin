from .hooke import *

def Hooke(mu, nu, dtype, dim):
    if dtype == np.float64:
        if dim == 2:
            return hooke.Hooke_2f64(mu, nu)
        elif dim == 3:
            return hooke.Hooke