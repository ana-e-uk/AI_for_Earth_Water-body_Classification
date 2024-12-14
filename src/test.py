'''

Script to test model outputs
'''

'''####################################################### Imports #######################################################''' 
import os
import numpy as np
import torch
from model import EncoderCNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, precision_score, recall_score
 
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


pred_array = np.load('pred_array.npy')
label_array = np.load('label_array.npy')

print("Arrays loaded successfully.")
print(pred_array.shape)
print(label_array.shape)
# print(classification_report(label_array, pred_array, digits=4))
# print(f1_score(y_true=label_array, y_pred=pred_array, average='macro'))

print(classification_report(label_array, pred_array, digits=4))
print("\nCONFUSION MATRIX:")
confusion_matrix_set = confusion_matrix(label_array,pred_array)
print(confusion_matrix_set)
f1_score_array = f1_score(label_array, pred_array, average = None)
precision_score_array = precision_score(label_array, pred_array, average = None)
recall_score_array = recall_score(label_array, pred_array, average = None)
# f1_score_array = f1_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# precision_score_array = precision_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
# recall_score_array = recall_score(label_array, pred_array, labels = config.labels_for_cl, average = None)
for r in range(f1_score_array.shape[0]):
    print(f1_score_array[r],end = ' ')
print('')
for r in range(precision_score_array.shape[0]):
    print(precision_score_array[r],end = ' ')
print('')
for r in range(recall_score_array.shape[0]):
    print(recall_score_array[r],end = ' ')
print('')
# Save f1 score, precision, and recall arrays
np.savetxt("f1_score.csv", f1_score_array, delimiter=",", fmt="%.4f")
np.savetxt("precision_score.csv", precision_score_array, delimiter=",", fmt="%.4f")
np.savetxt("recall_score.csv", recall_score_array, delimiter=",", fmt="%.4f")
print("F1 score, precision, and recall saved to 'f1_score.csv', 'precision_score.csv', and 'recall_score.csv'.")

# Save classification report
with open("classification_report.txt", "w") as f:
    f.write(classification_report(label_array, pred_array, digits=4))
print("Classification report saved to 'classification_report.txt'.")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_set, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save the figure
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("Confusion matrix plot saved as 'confusion_matrix.png'.")
