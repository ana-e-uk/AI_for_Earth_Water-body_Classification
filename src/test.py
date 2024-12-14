'''

Script to test model outputs
TODO: delete if not needed
'''

'''####################################################### Imports #######################################################''' 
import os
import numpy as np
import torch
from model import EncoderCNN

import config 
'''####################################################### Testing Encoder-Decoder #######################################################''' 

criterion = torch.nn.CrossEntropyLoss()

model = EncoderCNN(in_channels_spatial=config.channels, in_channels_temp= config.channels)
print('Created Model')
model = model.to(config.device)
model.load_state_dict(torch.load(os.path.join(config.model_dir_e_CNN, config.load_model_name_e_CNN+".pt")),strict = False)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
print('Created optimizer')
'''####################################################### Testing Encoder + CNN #######################################################''' 

loss = 0
preds = []
labels = []
IDs_all = []

for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(test_loader):
    
    optimizer.zero_grad()

    code_vec, out = model(image_patch_s.to(config.device).float(), image_patch_t.to(config.device).float())

    label_batch = label_batch.type(torch.long).to(config.device)
    batch_loss = criterion(out, label_batch)
    loss += batch_loss.item()

    out_label_batch = torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1)
    out_label_batch_cpu = out_label_batch.detach().cpu().numpy()
    label_batch_cpu = label_batch.detach().cpu().numpy()
        
    preds.append(out_label_batch_cpu)
    labels.append(label_batch_cpu)

    del out
    del code_vec

loss = loss/(batch+1)
print('Test Loss:{} '.format(loss), end="\n")
print("\n")

pred_array = np.concatenate(preds, axis=0)
label_array = np.concatenate(labels, axis=0)

print(pred_array.shape)
print(label_array.shape)
# Assuming pred_array and label_array are already defined
np.save('pred_array.npy', pred_array)
np.save('label_array.npy', label_array)

print("Arrays saved to 'pred_array.npy' and 'label_array.npy'.")