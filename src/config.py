'''
Constants for AI for earth
'''

'''####################################################### Imports #######################################################''' 


'''####################################################### Data #######################################################''' 
# spatial CNN
channels = 1
patch_size = 64

# spatial decoder
latent_dim = 256
output_size = patch_size
output_channels = channels

# temporal bi-directional LSTM
time_steps = 442

# other
num_classes = 4 # 5 when doing reservoir

'''####################################################### Model Params #######################################################''' 

