#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import config
import os
import numpy as np
import torch

from torchvision import models

torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
class SpatialAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAutoencoder, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, 256)

        self.upfc = torch.nn.Linear(256, 4096)
        self.unpool3 = torch.nn.ConvTranspose2d(64 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(16 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, out_channels, 3, padding=1)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x):
#         x = x.view(-1, channels*patch_size*patch_size)
#         x = self.dropout(x)
        x = x.view(-1, config.channels, config.patch_size, config.patch_size)

        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))
        fc = self.relu(self.fc(conv3.view(-1,4096)))

        code_vec = fc

        upfc = self.relu(self.upfc(code_vec))
        upconv3 = self.relu(self.upconv3_2(self.relu(self.upconv3_1(self.unpool3(upfc.view(-1,64,8,8))))))
        upconv2 = self.relu(self.upconv2_2(self.relu(self.upconv2_1(self.unpool2(upconv3)))))
        out = self.upconv1_2(self.relu(self.upconv1_1(self.unpool1(upconv2))))
        out = out.view(-1, config.channels, config.patch_size, config.patch_size)
        return code_vec, out

class SpatialAutoencoder_supervised(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAutoencoder_supervised, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, 256)
        
        self.out1 = torch.nn.Linear(256, 256)
        self.out2 = torch.nn.Linear(256, config.out_classes)  

        self.maxpool = torch.nn.MaxPool2d(2)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x):

        x = x.view(-1, config.channels, config.patch_size, config.patch_size)

        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))
        fc = self.relu(self.fc(conv3.view(-1,4096)))

        code_vec = fc
        
        out1_fc = self.relu((self.out1(code_vec.view(-1,256))))
        out2_fc = self.out2(out1_fc.view(-1,256))

        return code_vec, out2_fc
    
class tae(torch.nn.Module):
    def __init__(self, in_channels, code_dim, out_channels, device):
        super(tae,self).__init__()

        # PARAMETERS
        self.input_channels = in_channels
        self.code_dim = code_dim
        self.output_channels = out_channels
        self.device = device

        # LAYERS
        self.instance_encoder = torch.nn.Linear(in_features=self.input_channels, out_features=self.code_dim) # AE
        self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True) # AE
        self.temporal_decoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True) # AE
        self.instance_decoder = torch.nn.Linear(in_features=self.code_dim, out_features=self.input_channels) # AE
    #     self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        # GET SHAPES
        batch, window, _ = x.shape
#         print('batch',batch,'window',window)

        # OPERATIONS
        x_encoder = self.instance_encoder(x) # ENCODE
        _, x_encoder = self.temporal_encoder(x_encoder) # ENCODE
        code_vec = x_encoder[0].squeeze() # ENCODE
        out = torch.zeros(batch, window, self.input_channels).to(self.device) # DECODE
        input = torch.unsqueeze(torch.zeros_like(code_vec), dim=1) # DECODE
        h = x_encoder # DECODE
        for step in range(window): # DECODE
            input, h = self.temporal_decoder(input, h) # DECODE
            out[:,step] = self.instance_decoder(input.squeeze()) # DECODE

        return code_vec, out

    
    
class tae_supervised(torch.nn.Module):
    def __init__(self, in_channels, code_dim, out_channels, device):
        super(tae_supervised,self).__init__()

        # PARAMETERS
        self.input_channels = in_channels
        self.code_dim = code_dim
        self.output_channels = out_channels
        self.device = device

        # LAYERS
        self.instance_encoder = torch.nn.Linear(in_features=self.input_channels, out_features=self.code_dim) # AE
        self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True) # AE
        self.out1 = torch.nn.Linear(256, 256)
        self.out2 = torch.nn.Linear(256, config.out_classes)  
        self.relu = torch.nn.ReLU(inplace=True)

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        # GET SHAPES
        batch, window, _ = x.shape
#         print('batch',batch,'window',window)

        # OPERATIONS
        x_encoder = self.instance_encoder(x) # ENCODE
        _, x_encoder = self.temporal_encoder(x_encoder) # ENCODE
        code_vec = x_encoder[0].squeeze() # ENCODE
        out1_fc = self.relu((self.out1(code_vec.view(-1,256))))
        out2_fc = self.out2(out1_fc.view(-1,256))

        return code_vec, out2_fc    
    

class SpatialtemporalAutoencoder(torch.nn.Module):
    def __init__(self, in_channels_spatial, out_channels_spatial, in_channels_temp, out_channels_temp):
        super(SpatialtemporalAutoencoder, self).__init__()
        
        self.code_dim = config.code_dim
        self.device = config.device
        self.in_channels_temp = in_channels_temp
        
        self.conv1_1 = torch.nn.Conv2d(in_channels_spatial, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, self.code_dim)
        self.fc_s = torch.nn.Linear(self.code_dim, self.code_dim)
        self.fc_t = torch.nn.Linear(self.code_dim, self.code_dim)

        self.upfc = torch.nn.Linear(self.code_dim, 4096)
        
        self.unpool3 = torch.nn.ConvTranspose2d(64 , 64, kernel_size=2, stride=2)
        self.upconv3_1 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.upconv3_2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.unpool2 = torch.nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2)
        self.upconv2_1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.upconv2_2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.unpool1 = torch.nn.ConvTranspose2d(16 , 16, kernel_size=2, stride=2)
        self.upconv1_1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.upconv1_2 = torch.nn.Conv2d(16, out_channels_spatial, 3, padding=1)
        
        self.instance_encoder = torch.nn.Linear(in_features=in_channels_temp, out_features=self.code_dim)
        self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)
        self.temporal_decoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)
        self.instance_decoder = torch.nn.Linear(in_features=self.code_dim, out_features=out_channels_temp) 

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x_s,x_t):

        x_s = x_s.view(-1, config.channels, config.patch_size, config.patch_size)
        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x_s)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))
        enc_s = self.relu(self.fc(conv3.view(-1,4096)))
        
        enc_s = self.relu(self.fc_s(enc_s))
        
        enc_s_norm = torch.nn.functional.normalize(enc_s, p=2.0, dim=1, eps=1e-12)
        
        batch, window, _ = x_t.shape
#         print(x_t.shape)
        x_encoder = self.instance_encoder(x_t) # ENCODE
        _, x_encoder = self.temporal_encoder(x_encoder) # ENCODE
        enc_t = x_encoder[0].squeeze() # ENCODE
        
        enc_t = self.relu(self.fc_t(enc_t))

        enc_t_norm = torch.nn.functional.normalize(enc_t, p=2.0, dim=1, eps=1e-12)

        code_vec = (enc_s_norm + enc_t_norm)
        
        out_t = torch.zeros(batch, window, self.in_channels_temp).to(self.device) # DECODE
        input = torch.unsqueeze(torch.zeros_like(code_vec), dim=1) # DECODE
        h = x_encoder # DECODE
        for step in range(window): # DECODE
            input, h = self.temporal_decoder(input, h) # DECODE
            out_t[:,step] = self.instance_decoder(input.squeeze()) # DECODE

        upfc = self.relu(self.upfc(code_vec))
        upconv3 = self.relu(self.upconv3_2(self.relu(self.upconv3_1(self.unpool3(upfc.view(-1,64,8,8))))))
        upconv2 = self.relu(self.upconv2_2(self.relu(self.upconv2_1(self.unpool2(upconv3)))))
        out_s = self.upconv1_2(self.relu(self.upconv1_1(self.unpool1(upconv2))))
        out_s = out_s.view(-1, config.channels, config.patch_size, config.patch_size)
            
        return code_vec, out_s, out_t
    
class SpatialtemporalAutoencoder_supervised(torch.nn.Module):
    def __init__(self, in_channels_spatial, out_channels_spatial, in_channels_temp, out_channels_temp):
        super(SpatialtemporalAutoencoder_supervised, self).__init__()
        
        self.code_dim = config.code_dim
        self.device = config.device
        self.in_channels_temp = in_channels_temp
        
        self.conv1_1 = torch.nn.Conv2d(in_channels_spatial, 16, 3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.fc = torch.nn.Linear(4096, self.code_dim)
        self.fc_s = torch.nn.Linear(self.code_dim, self.code_dim)
        self.fc_t = torch.nn.Linear(self.code_dim, self.code_dim)
        
        self.out1 = torch.nn.Linear(256, 256)
        self.out2 = torch.nn.Linear(256, config.out_classes)  

        self.instance_encoder = torch.nn.Linear(in_features=in_channels_temp, out_features=self.code_dim)
        self.temporal_encoder = torch.nn.LSTM(input_size=self.code_dim, hidden_size=self.code_dim, batch_first=True)

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=0.1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self,x_s,x_t):

        x_s = x_s.view(-1, config.channels, config.patch_size, config.patch_size)
        conv1 = self.maxpool(self.relu(self.conv1_2(self.relu(self.conv1_1(x_s)))))
        conv2 = self.maxpool(self.relu(self.conv2_2(self.relu(self.conv2_1(conv1)))))
        conv3 = self.maxpool(self.relu(self.conv3_2(self.relu(self.conv3_1(conv2)))))
        enc_s = self.relu(self.fc(conv3.view(-1,4096)))
        
        enc_s = self.relu(self.fc_s(enc_s))
        
        enc_s_norm = torch.nn.functional.normalize(enc_s, p=2.0, dim=1, eps=1e-12)
        
        batch, window, _ = x_t.shape
#         print(x_t.shape)
        x_encoder = self.instance_encoder(x_t) # ENCODE
        _, x_encoder = self.temporal_encoder(x_encoder) # ENCODE
        enc_t = x_encoder[0].squeeze() # ENCODE
        
        enc_t = self.relu(self.fc_t(enc_t))

        enc_t_norm = torch.nn.functional.normalize(enc_t, p=2.0, dim=1, eps=1e-12)

        code_vec = (enc_s_norm + enc_t_norm)
        out1_fc = self.relu((self.out1(code_vec.view(-1,256))))
        out2_fc = self.out2(out1_fc.view(-1,256))

            
        return code_vec, out2_fc
    

if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    