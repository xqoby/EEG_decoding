import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from itertools import combinations
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset
from einops.layers.torch import Rearrange, Reduce
from lavis.models.clip_models.loss import ClipLoss
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from utils import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
from modules.Mamba import *
from torch.autograd import Variable
from modules.Caputo import *

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

criterion_cls = torch.nn.CrossEntropyLoss().cuda()
LongTensor = torch.cuda.LongTensor 
temperature = 0.07

    
def get_1d_sincos_pos_embed_fromlength(embed_dim, patch_length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = np.arange(patch_length, dtype=float)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    if cls_token:
        pos_embed = np.concatenate([emb,np.zeros([1, embed_dim])], axis=0)
    else:
        pos_embed = emb
    return pos_embed

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_sph_sincos_pos_embed(embed_dim, sph_coordination, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    sph_coordination = sph_coordination.reshape(2,-1)

    sph_the = sph_coordination[0]
    sph_phi = sph_coordination[1]
    # use half of dimensions to encode sph_theta
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, sph_the)  # (channel_number, D/2)
    # use half of dimensions to encode sph_phi
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, sph_phi)  # (channel_number, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (channel_number, D)

    if cls_token:
        pos_embed = np.concatenate([pos_embed,np.zeros([1, embed_dim])], axis=0)
    return pos_embed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Ensure d_model is even
        if d_model % 2 != 0:
            d_model += 1

        self.d_model = d_model
        self.max_len = max_len

        # Generate spherical positional embeddings
        pe = get_sph_sincos_pos_embed(d_model, np.arange(max_len), cls_token=False)
        self.pe = torch.tensor(pe, dtype=torch.float).unsqueeze(1)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [time_length, batch_size, channel]
        """
        batch_size = x.size(1)
        time_length = x.size(0)
        
        # Adjust positional embeddings for batch size
        pe = self.pe[:time_length, :, :].repeat(1, batch_size, 1)
        pe = pe.to(x.device)
        
        # Remove extra dimensions if d_model was increased
        if self.d_model > x.size(2):
            pe = pe[:, :, :x.size(2)]
        
        x = x + pe
        return x


class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        # print("src", src.shape) # torch.Size([250, 512, 63])
        output = self.transformer_encoder(src)
        output = output.permute(1, 2, 0)
        # print("src", src.shape)
        return output  # Change shape back to [batch_size, channel, time_length]
    


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1)),
            nn.AvgPool2d((1, 17), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   #torch.Size([1, 1, 63, 250])
        x = self.tsconv(x)
        # print("tsconv", x.shape)   #torch.Size([1, 40, 1, 36])
        x = self.projection(x)
        # print("projection", x.shape)  #torch.Size([1, 36, 40])
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1840, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

class SCFD(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(SCFD, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1) 
        self.VSS = VSSBlock(hidden_dim=250, drop_path=0.1, attn_drop_rate=0.1, d_state=16, expand=2.0, is_light_sr=False)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x):
        # print(x.shape) #torch.Size([256, 63, 250])
        x = self.attention_model(x)
        
        x = self.subject_wise_linear[0](x)
        x = self.VSS(x, (7, 9))
        eeg_embedding = self.enc_eeg(x)    
        out = self.proj_eeg(eeg_embedding)
        # print(out.shape) #torch.Size([256, 1024])
        return out  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
caputo = CaputoEncoder(input_size=250, lstm_size=512, lstm_layers=1, output_size=1024).to(device)
      
def train_model(eeg_model, img_model, dataloader, optimizer, device, text_features_all, img_features_all):
    eeg_model.train()
    img_model.train()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    save_features= True
    mse_loss_fn = nn.MSELoss()
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        # print(eeg_data.shape) torch.Size([512, 63, 250])
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        eeg_features = eeg_model(eeg_data).float()
        img_features = img_model(img_features).float()
        
        # print(eeg_features.shape) torch.Size([512, 1024])
        caputo_features = caputo(eeg_data)

        # 逐元素相加
        eeg_features = 0.5*caputo_features + 0.5*eeg_features
        
        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale
        
        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        regress_loss =  mse_loss_fn(eeg_features, img_features)
        contrastive_loss = img_loss
        # MSE = nn.MSELoss()
        # MSE_img_loss = MSE(eeg_features, img_features)
        # loss = img_loss + MSE_img_loss 
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)
        loss = alpha * img_loss + (1 - alpha) * text_loss
        # loss = (alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10)
        
        
        logits_eeg = logit_scale * eeg_features @ img_features.T
        logits_img = logits_eeg.t()
        labels1 = torch.arange(eeg_data.shape[0])
        labels1 = Variable(labels1.cuda().type(LongTensor))
        loss_eeg = criterion_cls(logits_eeg, labels1)
        loss_img = criterion_cls(logits_img, labels1)
        loss_infoNCE = (loss_eeg + loss_img) / 2
        
        
        loss = loss + loss_infoNCE
        
        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()
        
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)



def evaluate_model(eeg_model, img_model, dataloader, device, text_features_all, img_features_all, k):
    eeg_model.eval()
    img_model.eval()
    
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            eeg_features = eeg_model(eeg_data).float()
            img_features = img_model(img_features).float()

        
            logit_scale = eeg_model.logit_scale 
            # print(eeg_features.type, text_features.type, img_features.type)
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            regress_loss =  mse_loss_fn(eeg_features, img_features)
            contrastive_loss = img_loss
            # MSE = nn.MSELoss(reduction="none")
            # MSE_img_loss = MSE(labels, img_features)
            # loss = img_loss + MSE_img_loss 
            loss = img_loss*alpha + text_loss*(1-alpha)
            # loss = (alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10)

            
            logits_eeg = logit_scale * eeg_features @ img_features.T
            logits_img = logits_eeg.t()
            labels1 = torch.arange(eeg_data.shape[0])
            labels1 = Variable(labels1.cuda().type(LongTensor))
            loss_eeg = criterion_cls(logits_eeg, labels1)
            loss_img = criterion_cls(logits_img, labels1)
            loss_infoNCE = (loss_eeg + loss_img) / 2

            loss = loss + loss_infoNCE
            
            total_loss += loss.item()

            
            for idx, label in enumerate(labels):
                
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]
                
                if k==200:
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
 
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
                    
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(sub, current_time, eeg_model, img_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    logger.watch(img_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(config['epochs']):
        
        train_loss, train_accuracy, features_tensor = train_model(eeg_model, img_model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all)
        if (epoch +1) % 5 == 0:                    
            
            if config['insubject']==True:       
                os.makedirs(f"./models/contrast/{config['encoder_type']}/{sub}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/{config['encoder_type']}/{sub}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/contrast/across/{config['encoder_type']}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/across/{config['encoder_type']}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        
        test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
        _, v2_acc, _ = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2)
        _, v4_acc, _ = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4)
        _, v10_acc, _ = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10)
        _, v50_acc, v50_top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=50)
        _, v100_acc, v100_top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all,  k=100)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        # "train_loss": train_loss,
        # "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
    
    # model.load_state_dict(best_model_weights)
    
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    
    
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results

import datetime
def main():  
    config = {
        "data_path": "/root/autodl-tmp/EEG/EEG_Image_decode/datasets/THINGS/Preprocessed_data_250Hz",
        "project": "train_pos_img_text_rep",
        "entity": "sustech_rethinkingbci",
        "name": "lr=3e-4_img_pos_pro_eeg",
        "lr": 3e-4,
        "epochs": 50,
        "batch_size": 256,
        "logger": True,
        "insubject": True,
        "encoder_type": 'SCFD',
        "img_encoder": 'Proj_img'
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = config['data_path']
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05','sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")  

    for sub in subjects:                    
        # Re-initialize the models for each subject
        eeg_model = globals()[config['encoder_type']]()
        img_model = globals()[config['img_encoder']]()
        eeg_model.to(device)
        img_model.to(device)
        optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters(), img_model.parameters()), lr=config['lr'])            
        
        print(f'Processing {sub}: number of parameters:', sum([p.numel() for p in itertools.chain(eeg_model.parameters(), img_model.parameters())]))

        train_dataset = EEGDataset(data_path, subjects=[sub] if config['insubject'] else [], exclude_subject=sub if not config['insubject'] else None, train=True)
        test_dataset = EEGDataset(data_path, subjects=[sub] if config['insubject'] else [], exclude_subject=sub if not config['insubject'] else None, train=False)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, img_model, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=config['logger'])
        
        # Save results to a CSV file
        results_dir = f"./outputs/contrast/{config['encoder_type']}/{sub}/{current_time}"
        os.makedirs(results_dir, exist_ok=True)          
        results_file = f"{results_dir}/{config['encoder_type']}_{'cross_exclude_' if not config['insubject'] else ''}{sub}.csv"
        
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {results_file}')
            
if __name__ == '__main__':
    main()