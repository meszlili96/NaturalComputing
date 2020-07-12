# EvolutionaryGAN

The implementation is based on the paper [https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch](https://github.com/WANG-Chaoyue/EvolutionaryGAN-pytorch)
 
The organisation of the repository is the following:

- **EvolutionaryGAN** folder contains the cource code.
- **EvolutionaryGAN/scripts** folder contains the shell scripts that can be used to run GAN training.
- **Gamma comparison eval metrics** folder contains the plots of all evaluation metrics for simulated data saved for gammas which produced the best KDEs. These are the same plots as included into the report.
- **GANs comparison eval metrics** folder contains the plots of all evaluation metrics for simulated data  saved for the compared GANs. These are the same plots as included into the report.
- **literature** folder contains the literature we worked with.
- **Loss functions illustration.png** plot of the loss function used in the implementation made to ensure the correctness of the implementation.
- **Diversity score function visualisation.png** plot of the diversity score used in the implementation made to ensure the correctness of the implementation.
- **sim_distr.png** plots of PDEs and scatter plots for the sample of size 10000 for simulated data disctributions.

The results of all our experimenst can be found in [Google Drive folder](https://drive.google.com/drive/folders/1FXKlqFchWaDg0Ha_1QYMNOv1npS3F5Jr?usp=sharing). 
- **EGAN simulated data gamma tuning** contains the results of all runs made for gamma tuning for both 8 and 25 Gaussians datasets. Each folder for a separate run has the following set of files:
   - **epoch N.png** - a scatter plot of a generated sample of size 10000 after the epoch number N
   - **hq_rate.png** - a plot of high quality samples rate change per epoch. 
   - **hq_rate.npy** - saved values of high quality samples rate per epoch. Helpful to recreate the plot and place a few plot in one figure for comparison.
   - **jsd.png** - a plot of Jensen-Shannon divergence change per epoch.
   - **jsd.npy** - saved values of Jensen-Shannon divergence per epoch.
   - **x_stdev.png** - a plot of x standard deviation change per epoch.
   - **x_stdev.npy** - saved values of x standard deviation per epoch.
   - **y_stdev.png** - a plot of y standard deviation change per epoch.
   - **y_stdev.npy** - saved values of y standard deviation per epoch.
   - **rs_ll.png** - a plot of change of true sample average log-likelihood under approximated generator distribution per epoch.
   - **rs_ll.npy** - saved values of true sample average log-likelihood under approximated generator distribution per epoch.
   - **KDE** - KDE of the disctribution learned by generator obtained at the end of training.
   - **selected_mutations_stats** - the statistics of selected mutations splitted by training stages.
   - **train_summary.png** - a plot of generator and discriminator losses per epoch.
- **Simulated data experiments** - contains the reults of final experiments on simulated data. The saved plots and statistics is the same as for gamma tuning.
- **MNIST experiments** - contains the reults of final experiments on simulated data. Each folder for a separate run has the following set of files:
   - **epoch N.png** - a sample of generated images obtained after the epoch number N.
   - **train_summary.png** - a plot of generator and discriminator losses per epoch.
- **Celeba experiments** - contains the reults of final experiments on simulated data. The saved plots and statistics is the same as for MNIST.
   

