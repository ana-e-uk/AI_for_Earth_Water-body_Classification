'''
Script to evaluate the model
'''

'''####################################################### Imports #######################################################''' 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score, recall_score

import config
'''####################################################### Plots function #######################################################''' 

def eval_results(l, p, data_name):
    print(classification_report(l, p, digits=4))
    print("\nCONFUSION MATRIX:")
    confusion_matrix_set = confusion_matrix(l,p)
    print(confusion_matrix_set)
    f1_score_array = f1_score(l,p, average = None)
    precision_score_array = precision_score(l,p, average = None)
    recall_score_array = recall_score(l,p, average = None)

    for r in range(f1_score_array.shape[0]):
        print(f1_score_array[r],end = ' ')
    print('')
    for r in range(precision_score_array.shape[0]):
        print(precision_score_array[r],end = ' ')
    print('')
    for r in range(recall_score_array.shape[0]):
        print(recall_score_array[r],end = ' ')
    print('')

    # Save classification report
    with open(f"classification_report_{data_name}.txt", "w") as f:
        f.write(classification_report(l,p, digits=4))
    print(f"Classification report saved to 'classification_report_{data_name}.txt'.")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_set, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(config.RESULTS, f"confusion_matrix_{data_name}.png"), dpi=300)
    plt.close()


'''####################################################### Results testing data #######################################################''' 

pred_array = np.load('pred_array_e_CNN.npy')
label_array = np.load('label_array_e_CNN.npy')

print(f"\nTesting data from training region:")
eval_results(label_array, pred_array, "test")

'''####################################################### Results unseen data #######################################################''' 

u_pred_array = np.load('pred_array_e_CNN_unseen_data.npy')
u_label_array = np.load('label_array_e_CNN_unseen_data.npy')

print(f"\nTesting data from unseen region:")
eval_results(u_label_array, u_pred_array, "unseen")

'''#######################################################  #######################################################''' 