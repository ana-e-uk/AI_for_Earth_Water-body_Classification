'''

Script to test model outputs
'''

'''####################################################### Imports #######################################################''' 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, precision_score, recall_score
 
'''####################################################### Testing Encoder-Decoder #######################################################''' 



'''####################################################### Testing Encoder + CNN #######################################################''' 

<<<<<<< HEAD
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
=======

print("hello")
>>>>>>> 37587ba (Reconstruction visualization)
