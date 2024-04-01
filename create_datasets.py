# create a sweep of datasets for the XOR task with different levels of scaling
import numpy as np
import os

# Load original dataset
xor_data = np.load(os.path.join("data", "xor_train_data_no_scale.npz"))
train_inputs = xor_data["inputs"]
train_outputs = xor_data["outputs"]

I_pos = 0.33
I_neg = 0.11
I_0 = 0.45
L_0 = -0.087
n_trials = 11

# helper to scale values: as scale goes from 0 to 1, val should go from 1 to val
def scale(val, scale):
    return 1 + (val - 1) * scale

for i in range(n_trials):
    s = i / (n_trials - 1)

    train_inputs = []
    train_outputs = []

    for i1 in [0, 1]:
        for i2 in [0, 1]:
            train_inputs.append([I_neg*s, scale(I_pos, s), scale(I_0, s)*i1, scale(I_0, s)*i2])
            train_outputs.append([scale(L_0, s) * (i1 != i2)])

    train_inputs = np.array(train_inputs)
    train_outputs = np.array(train_outputs)

    np.savez(os.path.join("data", f"xor_train_data_scale_{s}.npz"), inputs=train_inputs, outputs=train_outputs)
    print(f"Saved dataset with scale {s}")