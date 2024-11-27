# Model Specifications

## Introduction
We will be implementing the model SLTLAE_CL so we can use the encoder weights as a starting point for the SLTLAE_CL_*enc_upd* model that we will use to classify farms, rivers, stable lakes, and seasonal lakes. We can use the scores listed in the paper for the SLTLAE_CL_*enc_upd* model (Table 2) as a baseline to determine if our model, call it M_2, impelementation is comparable with Ravirathinam et al..

Then, we will train the SLTLAE_CL model up to the SLTLAE_CL_*enc_upd* model again with an additional class (reservoirs) on the North American data, call this model M_2_r, and compare the results with the paper scores for the SLTLAE_CL_*enc_upd* model again. We expect the reservoir score to be similar to the other water-body scores. 

Finally, we will test the M_2_r  with reservoir data from different continents. This will let us compare how the model does with classifying the same water-body type in different locations.

## Model Description
Below is a description of the SLTLAE_CL model from [Ravirathinam et al.](https://www.researchgate.net/publication/364516705_Spatiotemporal_Classification_with_limited_labels_using_Constrained_Clustering_for_large_datasets).

### Multi-modal Autoencoder
*Defined in model.py*

This model's encoder $\mathcal{E}$ has a spatial and temporal component.

**Component Specifications:**

The spatial component is a CNN, while the temporal component is a bi-directional LSTM.

The final spatial embeddings $h^i_t$ are the output from the network's last layer. The final temporal embeddings $h^i_s$ are the sum of the forward and backward embeddings.

### Embedding Space
*Defined in model.py*

The spatial and temporal embeddings are combined into the multi-modal embeddings $h^i$ with the following steps:
1. Transformation function parameterized by the spatial and temporal parameters.
2. Activation function $\mathcal{F}$, ReLU
3. Normalization layer $\alpha$, $\mathcal{l}_{2}$

### Decoder
*Defined in model.py*

Both the spatial and temporal decoders take in the embeddings $h^i$.

The spatial decoder is a set of up-convolutional layers to reconstruct the fraction map.

The temporal decoder is an LSTM-based generator that reconstructs the time series. It iteratively outputs the data at each time based on the output data form the previous time steps.

### Training Loss
*Defined in model.py*

The training loss 

$$\mathcal{L} = \lambda \mathcal{L}^{Rec}_S + \gamma \mathcal{L}^{Rec}_L + \mathcal{L}^{Constrained}$$

is the mean-squared error between the reconstructed and original spatial and temporal inputs, plus the constrained loss between $h^i$ s from the same class. The equations in the paper are used for each of these parts. 

**Constrained Loss:** 
In order to deal with the class imbalance of the labels, we follow the paper's implementation details to create a subset of label pairs in each batch to use for the constrained loss.

The number of pairs is the minimum number of labels of all the classes present multiplied by 2 (where each label can be paired with itself).

### Implementation Paremeters
*Defined in model.py*

We use the final parameters listed in the implementation details of the paper for our model:

* batch size = 256
* $\lambda$ = 0.01
* $\gamma$ = 1
* number of epochs = 2000
* include the constrained loss from epoch 1000
* optimizer: Adam
* learning rate = 0.001

### Supervised Classifier
*Defined in model.py*

Once the Autoencoder-Decoder is trained, we will use the encoder weights to train the supervised classifier SLTLAE_CL_*enc_upd* by following the steps of the paper in section 4.2 and 4.3..
1. Choose 10 labels from each class, so 40 labels total for M_2 and 50 for M_2_r.
2. Use 40 labels by creating pairs and applying constraints, the rest of the data are unlabeled.
3. Add 2 fully connected layers at the end.

With the following parameters:

* batch size = 256 ?
* number of epochs = 50
* Cross entropy loss
* optimizer: Adam ?
* learning rate = 0.001 ?




