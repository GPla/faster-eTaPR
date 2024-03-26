# %%
import timeit

import numpy as np
from eTaPR_pkg import evaluate_w_streams
from faster_etapr import evaluate_from_preds

for n in [1_000, 10_000, 20_000]:
    print('-------------------')
    print(f'Running for n={n}')
    y = np.random.randint(0, 2, size=n)
    y_hat = np.random.randint(0, 2, size=n)
    theta_p, theta_r = 0.5, 0.1

    old_result = timeit.timeit(
        lambda: evaluate_w_streams(y_hat, y, theta_p=theta_p, theta_r=theta_r),
        number=1,
    )
    new_result = timeit.timeit(
        lambda: evaluate_from_preds(
            y_hat, y, theta_p=theta_p, theta_r=theta_r
        ),
        number=1,
    )
    print(
        f'eTaPR_pkg: {old_result:.4f}, faster_etapr: {new_result:.4f}, '
        f'({old_result / new_result}x)'
    )
