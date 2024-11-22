import sys
sys.path.append("../")
import config
import os
import numpy as np
import random

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, precision_score, recall_score
import time
import glob

import MODEL
import DATA_CODE.DATA_LOADER as DATA_LOADER

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if not os.path.exists(os.path.join(config.MODEL_DIR)):
    os.makedirs(os.path.join(config.MODEL_DIR))
print(config.MODEL_DIR)
if not os.path.exists(os.path.join(config.RESULTS_DIR)):
    os.makedirs(os.path.join(config.RESULTS_DIR))
if not os.path.exists(os.path.join(config.RESULTS_DIR,config.RESULT_FOLDER)):
    os.makedirs(os.path.join(config.RESULTS_DIR,config.RESULT_FOLDER))
print(config.MODEL_DIR)
create_pdf = 0
oversample = 1

def mse_loss(input_image, target, ignored_index, reduction):
#     print(input_image.shape,target.shape)
    mask = input_image == ignored_index
    out = (input_image[~mask]-target[~mask])**2
#     print(out)
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out
    
# %%
print("#######################################################################")
print("BUILD MODEL")
model = MODEL.SpatialtemporalAutoencoder(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
model = model.to('cuda')

criterion = torch.nn.MSELoss(reduction = 'none')
# criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
# %%
print("#######################################################################")
print("LOAD MODEL")
print(config.MODEL_DIR)
# print(os.listdir())
# print(os.listdir(os.path.join(MODEL_DIR + '/')))
model.load_state_dict(torch.load(os.path.join(config.load_MODEL_DIR, config.load_MODEL_NAME+".pt")))
print(config.load_MODEL_NAME)
model.eval()
# %%
print("#######################################################################")
print("TEST MODEL")
train_loss = []

image_patches_spatial_list_train = []
label_patches_spatial_list_train = []
image_patches_temp_list_train = []
label_patches_temp_list_train = []
label_IDs_list_train = []
IDs_list_train = []

paths_list = glob.glob(os.path.join(config.WARP_DIR,config.WARP_FOLDER, '*.npy'))
paths_label_list = []

conti_path_list = []

if(config.sub_conti == 1):
    continent_info = np.load('/home/kumarv/pravirat/Realsat_labelling/continent_info.npy')
    for path in paths_list:
        ID = path.split('/')[-1].split('_')[-4]
        if(continent_info[int(ID)] == config.continent_no):
            conti_path_list.append(path)

if(config.sub_conti == 1):
    paths_list = conti_path_list

ID_count_array = np.zeros((1000_000)).astype(np.int64)
path_dict = {}
for i,path in enumerate(paths_list):
#     print(path)
    ID = int(path.split('/')[-1].split('_')[1])
#     print(ID)
    ID_count_array[int(ID)] += 1
    path_dict[ID] = path

all_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/all_IDs_labels.npy')
print('Updated All label array  Shape:',all_label_array.shape, ' Count:',np.bincount(all_label_array)) 
print(len(paths_list))

train_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TRAIN))
print('TRAIN DATASET LENGTH:', train_dataset.shape)
labelled_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TRAIN_4CL_SAMESAMPLED))
print('WE HAVE LABELS FOR LENGTH:', labelled_dataset.shape)
labelled_dataset_list = labelled_dataset.tolist()

for ID_no in train_dataset:
    image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_no,all_label_array)
    
    if(len(image_patches_spatial_list_train) == config.no_train_images):
            break
            
    if(len(image_patches_spatial_list_train) % 1000 == 0):
            print(len(image_patches_spatial_list_train))
            
    image_patches_spatial_list_train.append(image_patches_spatial)
    label_patches_spatial_list_train.append(label_patches_spatial)    
    image_patches_temp_list_train.append(image_patches_temp)
    label_patches_temp_list_train.append(label_patches_temp)
    IDs_list_train.append(ID)
    
    if(ID in labelled_dataset_list):
        label_IDs_list_train.append(label_ID)
    else:
        label_IDs_list_train.append(0)
    
    # if(label_ID == 2 or label_ID == 6 or label_ID == 7 or label_ID == 8):
    #     label_IDs_list_train.append(0)
    # else:
    #     label_IDs_list_train.append(label_ID)

print(len(image_patches_spatial_list_train))
IDs_list_train_array = np.array(IDs_list_train)
label_IDs_list_train_array = np.array(label_IDs_list_train)
print(IDs_list_train_array.shape,label_IDs_list_train_array.shape,np.bincount(label_IDs_list_train_array))
# print(count_zero)

data_train = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list_train, label_patches_spatial_list_train,image_patches_temp_list_train,label_patches_temp_list_train,label_IDs_list_train,IDs_list_train)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=config.test_batch_size, shuffle=False, num_workers=0)

all_reps_train = []
train_outs_s = []
train_outs_t = []

for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader_train):

        code_vec, out_s, out_t= model(torch.tensor(image_patch_s).to('cuda').float(), image_patch_t.to('cuda').float())
        rep = code_vec.detach().cpu().numpy()
        out_s_cpu = out_s.detach().cpu().numpy()
        out_t_cpu = out_t.detach().cpu().numpy()
        print(rep.shape)
        
        for b in range(rep.shape[0]):
            all_reps_train.append(rep[b])
            train_outs_s.append(out_s_cpu[b])
            train_outs_t.append(out_t_cpu[b])

        del code_vec

print('No of images encoded: ',len(all_reps_train))

print(all_reps_train[1].shape)
num_clusters = config.num_clusters_model
print('No of clusters: ', config.num_clusters_model)

kmeans = KMeans(n_clusters=config.num_clusters_model, n_init=20).fit(all_reps_train)
# kmeans = KMeans(n_clusters=50, n_init=10).fit(all_reps_train)
# print(kmeans.cluster_centers_)
print(kmeans.cluster_centers_.shape)
train_rep_pred = kmeans.predict(all_reps_train)
transform_train_matrix = kmeans.transform(all_reps_train)
argsort_array = np.argsort(transform_train_matrix,axis = 1)

print(transform_train_matrix.shape)
print(transform_train_matrix[0,:])
print(argsort_array)
print(train_rep_pred)



            
print(config.load_MODEL_NAME)
num_clusters = config.num_clusters_model
print('No of clusters: ', config.num_clusters_model)

sup_train_ID_list = []
sup_train_ID_labels_list = []
add_to_new_train = 0
cluster_label_dict = {}
count_cluster_threshold = 2

for cluster_no in range(config.num_clusters_model):
# for cluster_no in range(10):
    add_to_new_train = 0
# for cluster_no in range(50):
    subset_ID_labels = label_IDs_list_train_array[train_rep_pred == cluster_no]
    subset_IDs = IDs_list_train_array[train_rep_pred == cluster_no]
    count_array = np.bincount(subset_ID_labels)
    
    if(count_array.shape[0] >= 2):
        
        subset_transform_matrix = transform_train_matrix[train_rep_pred == cluster_no]
        subset_transform_matrix_cluster = subset_transform_matrix[:,cluster_no]
        
        subset_labelled_IDs = subset_IDs[subset_ID_labels != 0]
        subset_labelled_IDs_labels = subset_ID_labels[subset_ID_labels != 0]
        subset_labelled_IDs_transform_matrix_cluster = subset_transform_matrix_cluster[subset_ID_labels != 0]
        mean_labelled_IDs_center_distance = np.mean(subset_labelled_IDs_transform_matrix_cluster)
        
        
        count_array_subset = count_array[1:]
        
        sub_count_argmax = np.argmax(count_array_subset)
        count_max = count_array_subset[sub_count_argmax]
        cluster_argsort_array = np.argsort(count_array_subset)
        
        if(cluster_argsort_array.shape[0] >= 2):
            
            sub_count_argmax_two = cluster_argsort_array[-2]
            count_second_max = count_array_subset[sub_count_argmax_two]
            
            if(count_max >= count_cluster_threshold and count_second_max<=0):
                add_to_new_train = 1
                
        else:
            if(count_max >= count_cluster_threshold):
                add_to_new_train = 1
        
        if(add_to_new_train == 1):
            count_added = 0
            for i in range(subset_IDs.shape[0]):
                if(subset_transform_matrix_cluster[i] < mean_labelled_IDs_center_distance):
                    count_added += 1
                    sup_train_ID_list.append(subset_IDs[i])
                    if(subset_ID_labels[i] == 0):
                        sup_train_ID_labels_list.append(sub_count_argmax + 1)
                    else:
                        sup_train_ID_labels_list.append(subset_ID_labels[i])
            print(cluster_no,count_array,'added',count_added)
            
        else:
            print(cluster_no,count_array)
            for i in range(subset_IDs.shape[0]):
                if(subset_ID_labels[i] != 0):
                    sup_train_ID_list.append(subset_IDs[i])
                    sup_train_ID_labels_list.append(subset_ID_labels[i])



print(len(sup_train_ID_list))
print(len(sup_train_ID_labels_list))
print(np.bincount(np.array(sup_train_ID_labels_list)))
og_count = np.bincount(np.array(sup_train_ID_labels_list))
# print(sup_train_ID_list)
# print(sup_train_ID_labels_list)
del model
# exit()
sup_train_ID_array = np.array(sup_train_ID_list)
sup_train_ID_labels_array = np.array(sup_train_ID_labels_list)

## OVERSAMPLE ##
if(oversample == 1):
    count_array = np.bincount(sup_train_ID_labels_array)
    max_count_label = np.amax(count_array)

    for l in range(count_array.shape[0]):
        subset_sup_train_ID = sup_train_ID_array[sup_train_ID_labels_array == l]
        subset_sup_train_ID_labels = sup_train_ID_labels_array[sup_train_ID_labels_array == l]
        no_of_labels = subset_sup_train_ID_labels.shape[0]

        if(no_of_labels != 0):
            for a in range(max_count_label - no_of_labels):
                rand_no = random.sample(range(no_of_labels), 1)
                sup_train_ID_list.append(subset_sup_train_ID[rand_no[0]])
                sup_train_ID_labels_list.append(subset_sup_train_ID_labels[rand_no[0]])


################################################################################
######################## TRAIN WITH NEW IDS AND LABELS #########################
################################################################################


print("#######################################################################")
print("BUILD MODEL")       
model = MODEL.SpatialtemporalAutoencoder_supervised(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
model= model.to('cuda')

if(config.load_model == 1):
    print("LOADING MODEL")
    print(config.load_MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(config.load_MODEL_DIR, config.load_MODEL_NAME+".pt")),strict = False)

            
for name, param in model.named_parameters():
    print(name,param.requires_grad)
    
criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# %%
print("#######################################################################")
print("TRAIN MODEL")
train_loss = []

image_patches_spatial_list = []
label_patches_spatial_list = []
image_patches_temp_list = []
label_patches_temp_list = []
label_IDs_list = []
IDs_list = []

# paths_list = glob.glob(os.path.join(config.PAD_DIR,config.PAD_FOLDER, '*.npy'))
paths_list = glob.glob(os.path.join(config.WARP_DIR,config.WARP_FOLDER, '*.npy'))

conti_path_list = []

if(config.sub_conti == 1):
    continent_info = np.load('/home/kumarv/pravirat/Realsat_labelling/continent_info.npy')
    for path in paths_list:
        ID = path.split('/')[-1].split('_')[-4]
        if(continent_info[int(ID)] == config.continent_no):
            conti_path_list.append(path)

if(config.sub_conti == 1):
    paths_list = conti_path_list

# paths_list = paths_list[random.sample(range(len(paths_list)), len(paths_list))]
print(len(paths_list))

ID_count_array = np.zeros((1000_000)).astype(np.int64)
path_dict = {}
for i,path in enumerate(paths_list):
#     print(path)
    ID = int(path.split('/')[-1].split('_')[1])
#     print(ID)
    ID_count_array[int(ID)] += 1
    path_dict[ID] = path

all_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/all_IDs_labels.npy')
print('Updated All label array  Shape:',all_label_array.shape, ' Count:',np.bincount(all_label_array))   

print('UPDATED DATASET LENGTH:', len(sup_train_ID_list))


for ID_no,ID_name in enumerate(sup_train_ID_list):
    
    image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_name,all_label_array)

    image_patches_spatial_list.append(image_patches_spatial)
    label_patches_spatial_list.append(label_patches_spatial)    
    image_patches_temp_list.append(image_patches_temp)
    label_patches_temp_list.append(label_patches_temp)
    label_IDs_list.append(sup_train_ID_labels_list[ID_no])
    IDs_list.append(ID)

print(len(image_patches_spatial_list))
# print(count_zero)
label_IDs_list_updated = []
for label_value in label_IDs_list:
    label_IDs_list_updated.append(config.label_dict[label_value])
                                  
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list)))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_updated)))

data = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list,label_patches_spatial_list,image_patches_temp_list,label_patches_temp_list,label_IDs_list_updated,IDs_list)
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size_labelled_only, shuffle=True, num_workers=0)

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []


for epoch in range(1,config.no_epochs_train+1):
    print('## EPOCH {} ##'.format(epoch))
    model.train()

    train_time_start = time.time()
    epoch_loss = 0
    
    for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader):
        
        optimizer.zero_grad()
        code_vec, out = model(image_patch_s.to('cuda').float(), image_patch_t.to('cuda').float())
    
        # label_patch_s = label_patch_s.to('cuda').float()
        # label_patch_t = label_patch_t.to('cuda').float()
        label_batch = label_batch.type(torch.long).to('cuda')

        batch_loss = criterion(out, label_batch)

        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        
    epoch_loss = epoch_loss/(batch+1)
    if(epoch_loss < 0.0005):
        break
    print(epoch_loss)
    print('\n')  
    train_loss.append(epoch_loss)

    model.eval()
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, str(config.experiment_id) + "_epoch_" + str(epoch) + "_train_loss_" + str("{:.4f}".format(train_loss[-1]))+".pt"))
    
    
    
test_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TEST))
print('TEST DATASET LENGTH:', test_dataset.shape)

image_patches_spatial_list_test = []
label_patches_spatial_list_test = []
image_patches_temp_list_test = []
label_patches_temp_list_test = []
label_IDs_list_test = []
IDs_list_test = []

for ID_no in test_dataset:
    # path = path_dict[int(ID_no)]
    if(all_label_array[int(ID_no)] != 0 and all_label_array[int(ID_no)] != 2 and all_label_array[int(ID_no)] != 6 and all_label_array[int(ID_no)] != 7 and all_label_array[int(ID_no)] != 8 and all_label_array[int(ID_no)] != 0):
        image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_no,all_label_array)
    #     print('Shapes', image_patches.shape,label_patches.shape,len(image_patches_list))

        if(len(image_patches_spatial_list_test) == config.no_train_images):
                break

        if(len(image_patches_spatial_list_test) % 1000 == 0):
                print(len(image_patches_spatial_list_test))

        image_patches_spatial_list_test.append(image_patches_spatial)
        label_patches_spatial_list_test.append(label_patches_spatial)
        image_patches_temp_list_test.append(image_patches_temp)
        label_patches_temp_list_test.append(label_patches_temp)
        label_IDs_list_test.append(label_ID)
        IDs_list_test.append(ID)

print(len(image_patches_spatial_list_test))
# print(np.bincount(np.array(label_IDs_list_test)))
# print(count_zero)
label_IDs_list_updated_test = []
for label_value in label_IDs_list_test:
    label_IDs_list_updated_test.append(config.label_dict[label_value])
                                  
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_test)))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_updated_test)))
                                  
data_test = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list_test, label_patches_spatial_list_test, image_patches_temp_list_test, label_patches_temp_list_test, label_IDs_list_updated_test, IDs_list_test)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=config.batch_size_labelled_only, shuffle=False, num_workers=0)

loss = 0
preds = []
labels = []
IDs_all = []

for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader_test):
    
    optimizer.zero_grad()

    code_vec, out = model(image_patch_s.to('cuda').float(), image_patch_t.to('cuda').float())

    label_batch = label_batch.type(torch.long).to('cuda')
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

print(classification_report(label_array, pred_array, digits=4))
print("\nCONFUSION MATRIX:")
confusion_matrix_set = confusion_matrix(label_array,pred_array)
print(confusion_matrix_set)
f1_score_array = f1_score(label_array, pred_array, average = None)
f1_nofreeze = f1_score_array
precision_score_array = precision_score(label_array, pred_array, average = None)
recall_score_array = recall_score(label_array, pred_array, average = None)
for r in range(f1_score_array.shape[0]):
    print(f1_score_array[r],end = ' ')
print('')
print(og_count)
print(np.bincount(np.array(sup_train_ID_labels_list)))
for r in range(precision_score_array.shape[0]):
    print(precision_score_array[r],end = ' ')
print('')
for r in range(recall_score_array.shape[0]):
    print(recall_score_array[r],end = ' ')
print('')
print(og_count)
print(np.bincount(np.array(sup_train_ID_labels_list)))

del model

################################################################################
#################### TRAIN WITH NEW IDS AND LABELS FREEZE ######################
################################################################################


print("#######################################################################")
print("BUILD MODEL")       
model = MODEL.SpatialtemporalAutoencoder_supervised(in_channels_spatial=config.channels, out_channels_spatial=config.channels,in_channels_temp= config.channels,out_channels_temp = config.channels)
model= model.to('cuda')


if(config.load_model == 1):
    print("LOADING MODEL")
    print(config.load_MODEL_NAME)
    model.load_state_dict(torch.load(os.path.join(config.load_MODEL_DIR, config.load_MODEL_NAME+".pt")),strict = False)

# if(config.freeze_layers == 1):
for name, param in model.named_parameters():
    if(name not in config.not_freeze_list):
        param.requires_grad = False
            
for name, param in model.named_parameters():
    print(name,param.requires_grad)
    
criterion = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# %%
print("#######################################################################")
print("TRAIN MODEL")
train_loss = []

image_patches_spatial_list = []
label_patches_spatial_list = []
image_patches_temp_list = []
label_patches_temp_list = []
label_IDs_list = []
IDs_list = []

# paths_list = glob.glob(os.path.join(config.PAD_DIR,config.PAD_FOLDER, '*.npy'))
paths_list = glob.glob(os.path.join(config.WARP_DIR,config.WARP_FOLDER, '*.npy'))

conti_path_list = []

if(config.sub_conti == 1):
    continent_info = np.load('/home/kumarv/pravirat/Realsat_labelling/continent_info.npy')
    for path in paths_list:
        ID = path.split('/')[-1].split('_')[-4]
        if(continent_info[int(ID)] == config.continent_no):
            conti_path_list.append(path)

if(config.sub_conti == 1):
    paths_list = conti_path_list

# paths_list = paths_list[random.sample(range(len(paths_list)), len(paths_list))]
print(len(paths_list))

ID_count_array = np.zeros((1000_000)).astype(np.int64)
path_dict = {}
for i,path in enumerate(paths_list):
#     print(path)
    ID = int(path.split('/')[-1].split('_')[1])
#     print(ID)
    ID_count_array[int(ID)] += 1
    path_dict[ID] = path

all_label_array = np.load('/home/kumarv/pravirat/Realsat_labelling/all_IDs_labels.npy')
print('Updated All label array  Shape:',all_label_array.shape, ' Count:',np.bincount(all_label_array))   

print('UPDATED DATASET LENGTH:', len(sup_train_ID_list))


for ID_no,ID_name in enumerate(sup_train_ID_list):
    
    image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_name,all_label_array)

    image_patches_spatial_list.append(image_patches_spatial)
    label_patches_spatial_list.append(label_patches_spatial)    
    image_patches_temp_list.append(image_patches_temp)
    label_patches_temp_list.append(label_patches_temp)
    label_IDs_list.append(sup_train_ID_labels_list[ID_no])
    IDs_list.append(ID)

print(len(image_patches_spatial_list))
# print(count_zero)
label_IDs_list_updated = []
for label_value in label_IDs_list:
    label_IDs_list_updated.append(config.label_dict[label_value])
                                  
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list)))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_updated)))

data = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list,label_patches_spatial_list,image_patches_temp_list,label_patches_temp_list,label_IDs_list_updated,IDs_list)
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=config.batch_size_labelled_only, shuffle=True, num_workers=0)

all_reps = []
valid_paths = []
valid_outs_s = []
valid_outs_t = []


for epoch in range(1,config.no_epochs_train+1):
    print('## EPOCH {} ##'.format(epoch))
    model.train()

    train_time_start = time.time()
    epoch_loss = 0
    
    for batch, [image_patch_s, label_patch_s,image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader):
        
        optimizer.zero_grad()
        code_vec, out = model(image_patch_s.to('cuda').float(), image_patch_t.to('cuda').float())
    
        # label_patch_s = label_patch_s.to('cuda').float()
        # label_patch_t = label_patch_t.to('cuda').float()
        label_batch = label_batch.type(torch.long).to('cuda')

        batch_loss = criterion(out, label_batch)

        batch_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()
        
    epoch_loss = epoch_loss/(batch+1)
    print(epoch_loss)
    print('\n')  
    train_loss.append(epoch_loss)

    model.eval()
    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, str(config.experiment_id) + "_epoch_" + str(epoch) + "_train_loss_" + str("{:.4f}".format(train_loss[-1]))+".pt"))
    
    
    
test_dataset = np.load(os.path.join(config.DATASET_DIR,config.DATASET_NAME_TEST))
print('TEST DATASET LENGTH:', test_dataset.shape)

image_patches_spatial_list_test = []
label_patches_spatial_list_test = []
image_patches_temp_list_test = []
label_patches_temp_list_test = []
label_IDs_list_test = []
IDs_list_test = []

for ID_no in test_dataset:
    # path = path_dict[int(ID_no)]
    if(all_label_array[int(ID_no)] != 0 and all_label_array[int(ID_no)] != 2 and all_label_array[int(ID_no)] != 6 and all_label_array[int(ID_no)] != 7 and all_label_array[int(ID_no)] != 8 and all_label_array[int(ID_no)] != 0):
        image_patches_spatial, label_patches_spatial, image_patches_temp, label_patches_temp, label_ID,ID = DATA_LOADER.get_data_spatial_temporal_label_ID_load_arrays(ID_no,all_label_array)
    #     print('Shapes', image_patches.shape,label_patches.shape,len(image_patches_list))

        if(len(image_patches_spatial_list_test) == config.no_train_images):
                break

        if(len(image_patches_spatial_list_test) % 1000 == 0):
                print(len(image_patches_spatial_list_test))

        image_patches_spatial_list_test.append(image_patches_spatial)
        label_patches_spatial_list_test.append(label_patches_spatial)
        image_patches_temp_list_test.append(image_patches_temp)
        label_patches_temp_list_test.append(label_patches_temp)
        label_IDs_list_test.append(label_ID)
        IDs_list_test.append(ID)

print(len(image_patches_spatial_list_test))
# print(np.bincount(np.array(label_IDs_list_test)))
# print(count_zero)
label_IDs_list_updated_test = []
for label_value in label_IDs_list_test:
    label_IDs_list_updated_test.append(config.label_dict[label_value])
                                  
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_test)))
print('LABEL COUNT:',np.bincount(np.array(label_IDs_list_updated_test)))
                                  
data_test = DATA_LOADER.SEGMENTATION_SLTL_PRED(image_patches_spatial_list_test, label_patches_spatial_list_test, image_patches_temp_list_test, label_patches_temp_list_test, label_IDs_list_updated_test, IDs_list_test)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=config.batch_size_labelled_only, shuffle=False, num_workers=0)

loss = 0
preds = []
labels = []
IDs_all = []

for batch, [image_patch_s, label_patch_s, image_patch_t, label_patch_t, label_batch, ID_batch] in enumerate(data_loader_test):
    
    optimizer.zero_grad()

    code_vec, out = model(image_patch_s.to('cuda').float(), image_patch_t.to('cuda').float())

    label_batch = label_batch.type(torch.long).to('cuda')
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

print(classification_report(label_array, pred_array, digits=4))
print("\nCONFUSION MATRIX:")
confusion_matrix_set = confusion_matrix(label_array,pred_array)
print(confusion_matrix_set)
f1_score_array = f1_score(label_array, pred_array, average = None)
f1_freeze = f1_score_array
precision_score_array = precision_score(label_array, pred_array, average = None)
recall_score_array = recall_score(label_array, pred_array, average = None)
for r in range(f1_score_array.shape[0]):
    print(f1_score_array[r],end = ' ')
print('')
for r in range(precision_score_array.shape[0]):
    print(precision_score_array[r],end = ' ')
print('')
for r in range(recall_score_array.shape[0]):
    print(recall_score_array[r],end = ' ')
print('')


print(og_count)
for r in range(f1_nofreeze.shape[0]):
    print(f1_nofreeze[r],end = ' ')
print('')
for r in range(f1_freeze.shape[0]):
    print(f1_freeze[r],end = ' ')
print('')
print(og_count)