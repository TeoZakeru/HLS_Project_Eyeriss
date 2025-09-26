from SystolicArray import EyerissF
import numpy as np

eyeriss = EyerissF()

np.random.seed(1)

picture = np.random.randint(0, 10, (5, 5), dtype=int)
filter_weight = np.random.randint(-5, 6, (3, 3), dtype=int)

output = eyeriss.Conv2d(picture, filter_weight, 1, 1)

print("Input shape:", picture.shape)
print("Filter shape:", filter_weight.shape)
print("Output shape:", output.shape)

print(picture)

print(filter_weight)

print(output)