# Data

## Intro
Our goal is to determine how much the model (SLTLAE_CL_*enc_upd*) is affected by spatial heterogeneity. 

Initially, we thought we could do this by:
* Training the model M_2 the same way as done in [Ravirathinam et al.](https://www.researchgate.net/publication/364516705_Spatiotemporal_Classification_with_limited_labels_using_Constrained_Clustering_for_large_datasets).
* Testing the performance of M_2 with the same data as done in Ravirathinam et al.. 
* Testing the performance of M_2 with data from a different location.

To do this, we need the following data:

1. Data $D_{o}$ used by Ravirathinam et al., which is the North American region.
2. Data $D_{n}$ from a different location that is as similar as possible to $D_{o}$ but is in a different spatial region. 

However, the data received did not have labels other than reservoirs for other continents. 

**Thus, our new plan is the following:**

1. Train the model M_2 the same way as done in Ravirathinam et al. (with data from North America only).
2. Test the performance of M_2 with the same data as done in Ravirathinam et al. (with labeled data from North America only).
3. Train a model (M_2_r) the same way as in the paper, but adding reservoir labeled data from North America.
4. Test M_2_r the same way as in the paper, but adding reservoir labeled data from North America, as well as reservoir data from all other contients.

This will let us explore how  M_2_r (and therefore SLTLAE_CL_*enc_upd*) performs in different continents when trained on one continent.

## Spatial Data
The spatial data has been preprocessed and given as a 64 x 64 x 1  fraction map with padding that allows all water body images to be square.

## Temporal Data
The temporal data has been preprocessed and given as a 442 x 1 time series.

## Data Needed

**North America Data:**
1. Training autoencoder-decoder for M_2 encoder weights:
    * All labels save for the ones needed for 2 and 3 (?)
    * 70 split (?) **(50 split of data)**
    * Divide into batch size of 256
    * For each batch, get 2*min*\{batch labeled data\} pairs of labeled data for constrained loss
2. Training model M_2 (supervised classifier):
    * 20 split (?)
    * 10 labels for each class of Farm, River, Stable Lake, Seasonal Lake
    * Divide into batch size of 256
    * For each batch ...
3. Testing model M_2 (supervised classifier):
    * 10 split (?)

4. Training autoencoder-decoder for M_2_r encoder weights:
    * All labels save for the ones needed for 5 and 6 (?)
    * 70 split (?)
    * Divide into batch size of 256
    * For each batch, get 2*min*\{batch labeled data\} pairs of labeled data for constrained loss
5. Training model M_2_r (supervised classifier):
    * 20 split (?)
    * 10 labels for each class of Farm, River, Stable Lake, Seasonal Lake, Reservoir
    * Divide into batch size of 256
    * For each batch, ...
6. Testing model M_2_r (supervised classifier):
    * 10 split (?)
    * 50 percent North America reservoir data, 50 percent unlabeled data

**All Continent Data:**
1. Testing model M_2_r (supervised classifier):
    * For each continent, take the same number of labels ($x$) that you had for North American Reservoirs for 6. (if there is not enough labels, just follow next bullet point)
    * For each continent, take $x$ unlabeled data, so 50 percent of the data is unlabeled, and 50 percent of the data is reservoir data

