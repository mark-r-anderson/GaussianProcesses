import numpy as np
from scipy.optimize import minimize

###################
#Gaussian processes
###################
            
class GPR:
    '''
    General class for performing Gaussian process regression.
    '''
    def __init__(self,kernel):
        self.kernel = kernel
        self.K = np.array([])
        self.K_inv = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        
    def train(self,x,y):
        '''
        Compute and invert the training covariance matrix and store the training data.
        '''
        #Store x and y
        self.x = x
        self.y = y

        #Compute the covariance matrix between the test data and itself
        self.set_K()
        
    def compute_K(self,x1,x2):
        '''
        Recompute K, K_inv and return the result.

        TODO:
        -add positive definite check
        '''
        #Compute the covariance matrix.
        K = self.kernel.get_cov_mat(x1,x2)

        #Compute the inverse of the covariance matrix.
        K_inv = np.linalg.inv(K)

        return K, K_inv

    def set_K(self):
        '''
        Recompute K, K_inv and set the attributes to the result.
        '''
        self.K, self.K_inv = self.compute_K(self.x,self.x)
        
    def predict(self,x_star,returnCov=True):
        '''
        Given a model and a set of independent observations, predict the corresponding dependent variable values.
        '''
        #Compute the covariance matrix between the test data and the training data.
        K_star = self.kernel.get_cov_mat(self.x,x_star)

        #Compute the covariance matrix between the training data and itself.
        K_star_star = self.kernel.get_cov_mat(x_star,x_star)
        
        #Compute the mean (each row of K_star adds another element to the array)
        y_star_mean = np.array( K_star * self.K_inv * np.transpose(np.matrix(self.y)) )
        
        #Remove the extra dimension for purposes of plotting later on (not really required, but convenient)
        y_star_mean = np.reshape(y_star_mean,(y_star_mean.size,))
    
        #Compute the variance by taking the diagonal elements
        y_star_cov = K_star_star - K_star * self.K_inv * np.transpose(K_star)
        y_star_var = np.diag(y_star_cov)
    
        if returnCov:
            #Return the mean and covariance matrix
            return y_star_mean, y_star_cov
        else:
            #Return the mean and variance
            return y_star_mean, y_star_var
    
    def sample_from_prior(self,x_star,n=1):
        '''
        Draw a desired number of samples from a Gaussian process prior.
        Based on the given kernel function. No training data.
        '''
        #Generate the mean for the prior distribution (assumed to be zero).
        m = np.transpose( np.matrix( np.zeros(x_star.size) ) )
        
        #Generate the covariance matrix for prior distribution.
        K = self.kernel.get_cov_mat(x_star)

        #Return n samples for distribution of N~(m,K)
        return self.get_samples(m,K,n)
    
    def sample_from_posterior(self,x_star,n=1):
        '''
        Draw a desired number of samples from a Gaussian process posterior.
        Based on the given kernel function and the traning data.
        '''
        #Generate the mean and the covariance matrix for the posterior distribution.
        m,K = self.predict(x_star)
        
        #Convert the mean to matrix format.
        m = np.transpose( np.matrix(m) )

        #Return n samples for distribution of N~(m,K)
        return self.get_samples(m,K,n)

    def optimize(self):
        '''
        Optimize over the hyperparameters.

        TODO:
        -Add argument to specify optimization method
        -Test this method way more to ensure it performs as expected
        -Ensure that set_hyperparameters updates the kernel hyperparameters as well
        '''
        #Set the initial hyperparameters to the ones entered by the user.
        hparams0 = self.kernel.get_hyperparameters()
        
        #Minimize the negative log marginal likelihood based on these initial parameters.
        res = minimize(self.lml,hparams0,method='Nelder-Mead', tol=1e-6)

        #Update the covariance matrices for the training data (test matrices updated in predict method).
        self.set_K()

        #If desired, can return the resulting object from SciPy optimize.
        return res

    def lml(self,hparams):
        '''
        Negative log marginal likelihood.

        TODO:
        -make hparams argument optional and use user-entered parameters instead
        '''
        #Reassign hyperparameters
        self.kernel.set_hyperparameters(hparams)

        #Covariance matrix must be recomputed and inverted every time!
        #K = self.kernel.get_cov_mat(self.x)
        #K_inv = np.linalg.inv(K)
        K,K_inv = self.compute_K(self.x,self.x)

        #Return the negative log marginal likelihood (scalar).
        return np.asscalar( np.matrix(self.y)*K_inv*np.transpose(np.matrix(self.y)) + np.linalg.det(K) )
    
    def get_samples(self,m,K,n=1):
        '''
        General method to sample from a distribution with mean m and covariance matrix K.
        Also accepts the number of samples desired.
        Returns n samples drawn from distribution of N~(m,K)

        TODO:
        -Fix square root
        -Add positive definite check and loop
        '''
        #Generate the identity matrix.
        I = np.matrix( np.identity( int(np.sqrt(K.size)) ) )
        
        #Add a bit of "noise" because the Cholesky decomposition may not work.
        K = self.numeric_fix(K)
        
        #Compute the Cholesky decomposition.
        L = np.matrix( np.linalg.cholesky(K) )
        
        #Generate random samples with mean **0** and covariance of the identity matrix (i.e., independent).
        u = np.transpose( np.matrix( np.diag( np.random.normal(0,I) ) ) )

        #Generate the desired number of samples.
        for i in range(1,n,1):
            u = np.append( u,np.transpose(np.matrix(np.diag(np.random.normal(0,I)))),axis=1 )

        #Return the sample(s) with distribution N~(m,K).
        return m + L*u

    def positive_definite(self,K):
        '''
        Check whether a matrix is positive definite.
        A matrix must be positive definite in order to compute the Cholesky decomposition.
        '''
        #For every eigenvalue.
        for eigval in np.linalg.eigvals(K):
            if eigval <= 0:
                #If an eigenvalue is not positive, then the matrix is not positive definite.
                return False

        #If every eigenvalue is positive, then the matrix is positive definite.
        return True

    def numeric_fix(self,K):
        '''
        Add a small multiple of the identity matrix to the covariance matrix.
        This can help to compute the Cholesky decomposition.
        '''
        #Define the identity matrix of appropriate size.
        I = np.matrix( np.identity( int(np.sqrt(K.size)) ) )

        #Define the multiple for the identity matrix
        epsilon = 1e-14
        
        #Define the maximum number of iterations until giving up.
        maxIter = int(1e9)

        #print(epsilon)
        #print(maxIter)        
        #print(K)

        for i in range(0,maxIter,1):
            if self.positive_definite(K):
                break
            K += epsilon*I
            #print("hi")
        
        #While the matrix is not positive definite.
        #while(not self.positive_definite(K)):
            #Add a small multiple of the identity matrix until it becomes positive definite.
            #K += 0.0000000000001*I

        #print(K)
            
        return K
        
