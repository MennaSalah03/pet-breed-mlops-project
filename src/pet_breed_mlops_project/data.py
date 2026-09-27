import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn  # noqa: PLR0402
import torch.nn as nn  # noqa: PLR0402
import torch.nn.functional as F
import torch.optim as optim  # noqa: PLR0402
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

cudnn.benchmark = True
plt.ion()   # interactive mode
# PyTorch idiom for using GPU if available, otherwise fallback to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
