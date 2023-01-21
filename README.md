Code for all experiments in the paper.

The `matrix approximation` folder contains the code for the kernel matrix approximation experiments.
1. The `genTestMatrix.py` provides methods that prepare the experiment data. 
2. The `bootstrap.py` implements the bootstrap procedure including the extraploation approach. 
3. The script `runs.py` has a complete list of Python commands that reproduce the experiment results.

The `krr + mmd` folder contains the code for both KRR and MMD experiments.
1. The `rffboot` folder is a reusable module that implements data generating methods, kernel related computations, and other util methods (such as plot).
2. The `kernel_ridge` folder has one script `main.py` and one implementation folder `impl`. Run the script file using `python main.py` to get the results of a KRR experiment. 
3. The `mmd` folder also has one script `main.py` and one implementation folder `impl`. Run the script file using `python main.py` to get the results of an MMD experiment.
4. Note: Both KRR and MMD experiments are parallel programs. When running the experiment, make sure that the machine can handle the experiment configurations.  