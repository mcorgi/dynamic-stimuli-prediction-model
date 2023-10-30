
import math 
from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib import animation
from torch.nn import MSELoss
import imageio
import os
import numpy as np
from functools import partial

from torchvision.transforms import Compose

# transformers stuff

import numpy as np
import torch

import transformers

from transformers import VivitImageProcessor, VivitModel, VivitForVideoClassification
from huggingface_hub import hf_hub_download

import time

from transformers import BitsAndBytesConfig

import accelerate
import bitsandbytes

import wandb
import random
import yaml
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

import sys

import torch
sys.path.append('sensorium_2023')
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.utility.scores import get_correlations
from sensorium.models.make_model import make_video_model

from bitsandbytes.nn import Linear4bit
from torch.nn import Linear
import torch
import torch.nn as nn
from transformers import VivitImageProcessor, VivitModel, VivitForVideoClassification
from transformers.modeling_outputs import ImageClassifierOutput
from utils import load_yaml, print_trainable_parameters, concatenate_dataloaders

DATA_PATHS = [
'/storage/sensorium/data/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20/',
# '/storage/sensorium/data/dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20/',
# '/storage/sensorium/data/dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20/',
# '/storage/sensorium/data/dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20/',
# '/storage/sensorium/data/dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20/'
]

print("Loading data..")
data_loaders = mouse_video_loader(
    paths=DATA_PATHS,
    batch_size=1,
    scale=1,
    max_frame=None,
    frames=64, # frames has to be > 50. If it fits on your gpu, we recommend 150
    offset=-1,
    include_behavior=True,
    include_pupil_centers=True,
    cuda=device!='cpu',
    to_cut=False,
)
print('Data loaded')

MICE = sorted(list(data_loaders['train'].keys()))

data_loaders_transpose = {}

for stage, stage_dataloaders in data_loaders.items():
    for mouse, dataloader in stage_dataloaders.items():
        if mouse not in data_loaders_transpose:
            data_loaders_transpose[mouse] = {stage: dataloader}
        else:
            data_loaders_transpose[mouse][stage] = dataloader

MOUSE_SIZES = {}
for mouse in MICE:
    batch = next(iter(data_loaders_transpose[mouse]['train']))
    MOUSE_SIZES[mouse] = batch.responses.shape[1]
MOUSE_SIZES

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

WINDOW_SIZE = 32

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

model = SensoriumVivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", quantization_config=bnb_config)

# model.classifier= Linear(768, 38000, compute_dtype=torch.bfloat16).cuda()


# model = VivitWrapper(model)

model.classifier = Swappable({mouse: Linear(768, output_size*WINDOW_SIZE) for mouse, output_size in MOUSE_SIZES.items()})


model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, config)
# train the classifier layer
for param in model.classifier.parameters():
    # classifier.register_forward_hook(make_module_require_grad)
    param.requires_grad_(True)
# for param in model.reducer.parameters():
#     # classifier.register_forward_hook(make_module_require_grad)
#     param.requires_grad_(True)
print_trainable_parameters(model)
model.to('cuda:0')

assert next(model.classifier.parameters()).requires_grad
assert next(model.classifier.parameters()).device == model.device

optimizer = bitsandbytes.optim.AdamW(model.parameters(),lr=3e-5,optim_bits=8, is_paged=True)

loss_fn = MSELoss()

max_length = 0
max_mouse = None
mice_lengths = []
mice_lengths_train = []
for stage, stage_dataloaders in data_loaders.items():
    for mouse, dataloader in stage_dataloaders.items():
        for batch in dataloader:
            length = batch.responses.shape[1]
            mice_lengths.append((length, stage, mouse))
            if 'train' in stage:
                mice_lengths_train.append((length, stage, mouse))
            if length > max_length:
                max_length = length
                max_mouse = (stage, mouse)
            break
print(max_length, max_mouse)
print(sorted(mice_lengths, key=lambda elt: elt[0], reverse=True))
print(sorted(mice_lengths_train, key=lambda elt: elt[0], reverse=True))

model = model.to('cuda:0')

DEBUG = False


best_val_loss = math.inf
save_period = 1
val_period = 1
n_epochs = 3
checkpoint_dir = Path("checkpoints")/str(datetime.now()).replace(" ", "_")
checkpoint_dir.mkdir(parents = True)
train_total = sum([len(loader) for loader in data_loaders['train'].values()])
print('total batches', train_total)
steps = 0
for epoch in range(1,n_epochs+1):
    print('EPOCH', epoch)
    model.train()
    for mouse, batch in tqdm(concatenate_dataloaders(data_loaders['train']), total=train_total):
        # model.classifier.to('cpu')
        # model.classifier = classifiers[mouse].to('cuda:0')
        # for batch in tqdm(data_loaders['train'][mouse]):
            # print(batch.videos)
        start = random.randint(50, 150-64)
        videos = batch.videos[:,:,start:start+64:2,:,:].permute(0,2,3,4,1)
        video_range = videos.max()-videos.min()
        videos -= videos.max() - video_range/2
        videos /= video_range + 0.01
        videos += .5
        videos = videos.squeeze()
        videos = image_processor(list(videos), return_tensors="pt")
        videos['pixel_values'] = videos['pixel_values'].cuda()
        responses = batch.responses[:,:,start:start+64:2].cuda()

        optimizer.zero_grad()
        output = model(**videos, mouse=mouse)
        # print(output.logits)
        train_loss= loss_fn(output.logits.reshape(-1, MOUSE_SIZES[mouse], WINDOW_SIZE), torch.log(torch.nn.functional.relu(responses)+1)) #torch.log(batch['responses'])
        # print(train_loss)
        wandb.log({'train_loss':train_loss.item()})
        train_loss.backward()
        optimizer.step()
        steps += 1
        if DEBUG:
            break
        if steps % 500 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_dir/f"step_{steps}.pt")
            
    
    if steps % save_period == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_dir/f"{epoch}.pt")

    if epoch%val_period==0:
        total_val_loss = 0

        for mouse in data_loaders['oracle']:
            # model.classifier.to('cpu')
            # model.classifier = classifiers[mouse].to('cuda:0')
            model.eval()
            for batch in data_loaders['oracle'][mouse]:
                start = random.randint(50, 150-64)
                videos = batch.videos[:,:,start:start+64:2,:,:].permute(0,2,3,4,1)
                video_range = videos.max()-videos.min()
                videos -= videos.max() - video_range/2
                videos /= video_range + 0.01
                videos += .5
                videos = videos.squeeze()
                videos = image_processor(list(videos), return_tensors="pt")
                videos['pixel_values'] = videos['pixel_values'].cuda()
                responses = batch.responses[:,:,start:start+64:2].cuda()

                with torch.no_grad():
                    output = model(**videos, mouse=mouse)
                    val_loss= loss_fn(output.logits.reshape(-1, MOUSE_SIZES[mouse], WINDOW_SIZE), torch.log(torch.nn.functional.relu(responses)+1)) #torch.log(batch['responses'])          
                total_val_loss+=val_loss
                if DEBUG:
                    break
            wandb.log({'val_loss':total_val_loss.item()})
            print("train_loss", train_loss.item())
            print("val loss:", total_val_loss.item())
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': total_val_loss,
                }, checkpoint_dir/"best.pt")
                print(f"save checkpoint to {checkpoint_dir/'best.pt'}")