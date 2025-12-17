import pylfire
from pylfire.models import arch
import numpy as np
import matplotlib

m = arch.get_model()
n=10
t1 = np.linspace(-1, 1, n)
t2 = np.linspace(0, 1, n)
tt1, tt2 = np.meshgrid(t1, t2, indexing='ij')
params_grid = np.c_[tt1.flatten(), tt2.flatten()]
batch_size = 100
lfire_method=pylfire.LFIRE(model=m, params_grid=params_grid, batch_size=batch_size)
lfire_res = lfire_method.infer()
lfire_res.summary()
lfire_res.plot_marginals()
lfire_res.results