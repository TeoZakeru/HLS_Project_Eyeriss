# import sys
# sys.path.append('../')
# from src.Hive import Hive
# from src.IO2 import RLE
# from src.Eyeriss import Eyeriss
# import numpy as np
# import skimage.io as io
# # from skimage.util import pad
# import os
# from model.lenet import LeNet5
# import torch
# from scipy import stats

# net = LeNet5()

# for i, (name, param) in enumerate(net.named_parameters()):
#     # print(name)
#     data = np.load("../filter/"+name+".npy")
#     param.data = torch.from_numpy(data)
# net.eval()
# dir_name = "../mnist_png/mnist_png/testing/7"
# #dir_name = "mnist_png/mnist_png/one_pic"
# files = os.listdir(dir_name)
# batch_size = 20
# r = RLE(1)
# e = Eyeriss()
# hive = Hive(e)
# total_samples = 0
# correct_predictions = 0

# # Extract ground truth label from folder name
# true_label = int(dir_name[-1])  # e.g., "7"
# for f in range(0, len(files), batch_size):
#     pics = []
#     for i in range(batch_size):
#         load_from = os.path.join(dir_name,files[f+i])
#         image = io.imread(load_from, as_gray=True)
#         image = np.pad(image,((2,2),(2,2)), 'median')
#         pic = np.array(image/255).reshape(1,image.shape[0],-1)
#         pics.append(pic) 
#     pics=np.array(pics)
#     inputs=torch.tensor(pics,dtype=torch.float32)
    
  
#     flts = np.load("../filter/convnet.c1.weight.npy")
#     pics = r.Compress(pics)
#     flts = r.Compress(flts)
#     pics= hive.Conv2d(pics,flts)
    
#     #pics = np.swapaxes(pics,1,3)
#     #pics= pics+np.float16(np.load("filter/convnet.c1.bias.npy"))
#     #pics = np.swapaxes(pics,1,3)
#     pics = hive.PreProcess(pics)
#     pics = hive.ReLU(pics)
#     pics=hive.Pooling(pics)
    
#     flts = np.float16(np.load("../filter/convnet.c3.weight.npy"))
#     flts = r.Compress(flts)
#     pics = r.Compress(pics)
#     #print('pic', pics.shape,'flt', flts.shape)
#     pics = hive.Conv2d(pics,flts)
#     #pics = np.swapaxes(pics,1,3)
#     #pics= pics+np.float16(np.load("filter/convnet.c3.bias.npy"))
#     #pics = np.swapaxes(pics,1,3)
#     #pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
#     pics = hive.PreProcess(pics)
#     pics = hive.ReLU(pics)
#     pics=hive.Pooling(pics)
    
    
    
    
    
#     #print('after pooling pic', pics.shape)
    
#     flts = np.float16(np.float16(np.load("../filter/convnet.c5.weight.npy")))
#     flts = r.Compress(flts)
#     pics = r.Compress(pics)
#     #print('pic', pics.shape,'flt', flts.shape)
#     pics = hive.Conv2d(pics, flts) 
    
#     #pics = np.swapaxes(pics,1,3)
#     #pics= pics+np.float16(np.load("filter/convnet.c5.bias.npy"))
#     #pics = np.swapaxes(pics,1,3)
#     #pics = Extension.NumpyAddExtension(hive.Decompress(r)) 
#     pics = hive.PreProcess(pics)
#     pics = hive.ReLU(pics)

    
#     res = inputs
#     for i in range(8):
#         res = net.convnet[i](res)
#     # print(res.shape)
#     diff = pics-res.data.numpy()
#     for i in range(len(res)):
#         for j in range(len(res[0])):
#             if np.any(np.abs(diff[i][j])>10e-4): 
#                 # print(diff[i][j])
#                 pass
#     #break
    
    
#     vector = pics.reshape(batch_size, -1)
#     vector = hive.FullConnect(vector, np.load('../filter/fc.f6.weight.npy'))
#     vector = hive.ReLU(vector)
#     #vector = vector+np.float16(np.load("filter/fc.f6.bias.npy"))
#     vector = hive.FullConnect(vector, np.load('../filter/fc.f7.weight.npy'))
#     #vector = vector+np.float16(np.load("filter/fc.f7.bias.npy"))

#     print("this number is : ",vector.argmax(axis = 1))
#     # for i in vector:
#     #     print(i)
#     # break



import sys
sys.path.append('../')
from src.Hive import Hive
from src.IO2 import RLE
from src.Eyeriss import Eyeriss
import numpy as np
import skimage.io as io
import os
from model.lenet import LeNet5
import torch
import random

# =====================
# Load trained LeNet5 model weights
# =====================
net = LeNet5()
for i, (name, param) in enumerate(net.named_parameters()):
    data = np.load("../filter/" + name + ".npy")
    param.data = torch.from_numpy(data)
net.eval()

# =====================
# Initialize accelerator simulation components
# =====================
r = RLE(1)
e = Eyeriss()
hive = Hive(e)
batch_size = 100  # use smaller batches for random sample

# =====================
# Sampling setup
# =====================
base_dir = "../mnist_png/mnist_png/testing"
all_files = []

# Collect all (image_path, label) pairs
for digit in range(10):
    dir_name = os.path.join(base_dir, str(digit))
    files = os.listdir(dir_name)
    for f in files:
        all_files.append((os.path.join(dir_name, f), digit))

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Randomly select 2000 samples
sampled_files = random.sample(all_files, 200)

total_correct = 0
total_samples = 0

print(f"\n--- Evaluating randomly selected 2000 MNIST test images ---")

# =====================
# Batch processing
# =====================
for f in range(0, len(sampled_files), batch_size):
    pics = []
    labels = []
    batch = sampled_files[f : f + batch_size]

    # Load batch
    for path, label in batch:
        image = io.imread(path, as_gray=True)
        image = np.pad(image, ((2, 2), (2, 2)), 'median')
        pic = np.array(image / 255).reshape(1, image.shape[0], -1)
        pics.append(pic)
        labels.append(label)

    pics = np.array(pics)
    inputs = torch.tensor(pics, dtype=torch.float32)

    # =====================
    # Eyeriss-simulated LeNet forward pass
    # =====================

    # Conv Layer 1
    flts = np.load("../filter/convnet.c1.weight.npy")
    pics = r.Compress(pics)
    flts = r.Compress(flts)
    pics = hive.Conv2d(pics, flts)
    pics = hive.PreProcess(pics)
    pics = hive.ReLU(pics)
    pics = hive.Pooling(pics)

    # Conv Layer 2 (C3)
    flts = np.float16(np.load("../filter/convnet.c3.weight.npy"))
    flts = r.Compress(flts)
    pics = r.Compress(pics)
    pics = hive.Conv2d(pics, flts)
    pics = hive.PreProcess(pics)
    pics = hive.ReLU(pics)
    pics = hive.Pooling(pics)

    # Conv Layer 3 (C5)
    flts = np.float16(np.load("../filter/convnet.c5.weight.npy"))
    flts = r.Compress(flts)
    pics = r.Compress(pics)
    pics = hive.Conv2d(pics, flts)
    pics = hive.PreProcess(pics)
    pics = hive.ReLU(pics)

    # Fully Connected Layers (F6, F7)
    vector = pics.reshape(pics.shape[0], -1)
    vector = hive.FullConnect(vector, np.load('../filter/fc.f6.weight.npy'))
    vector = hive.ReLU(vector)
    vector = hive.FullConnect(vector, np.load('../filter/fc.f7.weight.npy'))

    # =====================
    # Predictions and accuracy
    # =====================
    preds = vector.argmax(axis=1)
    labels = np.array(labels)
    total_correct += np.sum(preds == labels)
    total_samples += len(preds)

# =====================
# Final overall accuracy
# =====================
accuracy = total_correct / total_samples
print("\n===========================")
print(f"Accuracy on 2000 Random Samples: {accuracy * 100:.2f}%")
print("===========================")
