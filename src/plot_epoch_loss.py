'''
Script for plotting epoch loss
'''

'''####################################################### Imports #######################################################''' 
import numpy as np
import matplotlib.pyplot as plt

# Load epoch losses
# CHOOSE WHICH FILE TO PLOT
model = '_e_d'
# model = '_e_CNN'

# losses_file = f'epoch_losses{model}.npy'
losses_file = 'old_loss_epoch_losses_e_d.npy'

try:
    epoch_losses = np.load(losses_file)
except FileNotFoundError:
    print(f"Error: File '{losses_file}' not found. Make sure the file exists in the specified path.")
    exit(1)


for i in range(len(epoch_losses)):
    # if i ==300:
    #     break
    print(f"{i}\t{epoch_losses[i]}")


# Ensure epoch_losses is a 1D array
if epoch_losses.ndim != 1:
    print("Error: Expected 'epoch_losses' to be a 1D array. Check the file content.")
    exit(1)

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label='Training Loss', marker='o', linestyle='-', color='b')
plt.title('Epoch Losses of Encoder-Decoder Model', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

output_file = f'epoch_losses_plot_{model}.png'
plt.savefig(output_file)
print(f"Plot saved as '{output_file}'")
