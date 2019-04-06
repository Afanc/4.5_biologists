#!/usr/bin/python

import torch


print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)