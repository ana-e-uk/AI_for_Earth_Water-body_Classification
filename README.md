# AI_for_Earth_Water-body_Classification

To understand the available water and water systems in the world, it is useful to classify water-bodies into categories that can give us some information regarding their characteristics.

Our project aims to classify the water-bodies identified in the RealSAT dataset using Machine Learning models, following on the work done by [Ravirathinam et al](https://www.researchgate.net/publication/364516705_Spatiotemporal_Classification_with_limited_labels_using_Constrained_Clustering_for_large_datasets)

This repository contains the code related to the UMN CSCI 8523: AI for Earth class project.


# File Contents

* **data/**: contains dataset locally, and a README file with instructions on how the data can be obtained.

* **docs/**: documentation for the project.

* **src/**: contains all source code.

    * **code_reference/**: code used as reference for source code.

    * **load_data.py**: code for handling data loading, transformations, preprocessing.

    * **model.py**: code for defining and training models.

    * **utils/**: utility functions such as logging, metrics calculations, etc.

    * **train.py**: script to run the training loop.

    * **eval.py**: scrit to evaluate the model.

    * **predict.py**: script for inference on new data.

* **explore/**: scripts for experiments, exploration, etc..

* **init_venv_{ay, u, yo}.sh**: shell script for everyone's virtual environments.

* **README.md** (You are here): Overview of project, how to set it up, any usage instructions.

* **requirements.txt** (TO ADD): Lists the code dependencies.