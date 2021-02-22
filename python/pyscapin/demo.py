import numpy as np

import hooke
import ms94

if __name__ == "__main__":
    print(hooke.__author__)
    print(hooke.__version__)
    μ = 1.0
    ν = 0.3
    hooke = hooke.Hooke_3c128(μ, ν)
    print(hooke)
    print(type(hooke))
    print(hooke.dtype)
    # N = np.array((8, 16, 32), dtype=np.int)
    # L = np.array((1., 2., 4.), dtype=np.float64)
    # green = scapin.ms94.MoulinecSuquet94HookeComplex128_3D(hooke, N, L)
    # print(green)
    # print(green.gamma.mu)
    # print(green.gamma.nu)
    # print(green.N)
    # print(green.L)
