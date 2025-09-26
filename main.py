from SystolicArray import EyerissF
import numpy as np

eyeriss = EyerissF()

np.random.seed(2)

picture = np.random.randint(0, 255, (18, 18), dtype=int)
filter_weight = np.random.randint(-5, 6, (5, 5), dtype=int)

output = eyeriss.Conv2d(picture, filter_weight, 1, 1)

print("Input shape:", picture.shape)
print("Filter shape:", filter_weight.shape)
print("Output shape:", output.shape)

print(picture)

print(filter_weight)


print(output)

