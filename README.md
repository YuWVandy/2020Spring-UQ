##2020 Spring Semester Uncertainty Quantification Homework  
##----------------------------------------------------------  
Hw1: Generating random process using Spectral Simulation and KL expansion  
  
##----------------------------------------------------------  
Hw2: Polynomial Chao Estimation  
(1) Simulating random process of Yang's modules using KL expansion  
(2) Sampling training points from the random process  
(3) Calculating elongation using Physics model  
(4) Training the PCE model with training points  
(5) Compare the prediction results with physics model  
  
##----------------------------------------------------------  
Hw3: Gaussian Random Process Regression  
(1) Sampling training points from the random process generated during Hw2  
(2) Training the linear trend function using Maximum likelihood method  
(3) Training the noise part using gaussian random process regression  
(4) Compare the results with the physics model, PCE model, least square model  
(5) Compare the results with using current existed Python package  
  
##----------------------------------------------------------  
Hw4: Metroplis Algorithm  
(1) Sampling training points from the random process generated during Hw2  
(2) Calculate the PDF value for the sampling distribution and target distribution  
(3) Calculate the accepted ratio for each sample  
(4) Randomly generate a number from the uniform distribution (0, 1)
(5) Determine whether the training point is accepted and if so, add them to the current data pool  
(6) Build the PDF distribution of the data pool  
(7) Calculate the KL divergence value beween the current data pool and the initial sampling distribution for each variable  
(8) Determine whether the KL divergence value is stable, if so the state of each variable is stable  
  
##----------------------------------------------------------  
Hw5: Particle Filtering  
(1) Sampling training points from the prior uniform distribution for 5 variables  
(2) Sampling 5 independent points for physics model to calculate the observations  
(3) Calculating the likelihood for training points (Normal distribution with observation as the mean value, the last variable as the std)  
(4) Treating the likelihood as the PMF, calculating the CDF value for each training point  
(5) Using the CDF value as the updated prior to do sampling in the new iteration  