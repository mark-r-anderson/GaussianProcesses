import numpy as np

from scipy.optimize import minimize, rosen, rosen_der

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
        K = self.kernel.get_cov_mat(x1,x2)
        K_inv = np.linalg.inv(K)

        return K, K_inv

    def set_K(self):
        self.K, self.K_inv = self.compute_K(self.x,self.x)
        
    def predict(self,x_star,returnCov = True):
        '''
        Given a model and a set of independent observations, predict the corresponding dependent variable values.
        '''
        #Compute the covariance matrix between the test data and the training data
        #Each row corresponds to one test point
        K_star = self.kernel.get_cov_mat(self.x,x_star)

        #Compute the covariance matrix between the training data and itself
        #(each diagonal element is the covriance matrix for a single test point)
        K_star_star = self.kernel.get_cov_mat(x_star,x_star)

        #print(K_star * self.K_inv)
        #print(self.y)
        #print(K_star * self.K_inv * self.y[:,np.newaxis])
        #print('')
        #print(np.matrix(self.y))
        #print(K_star * self.K_inv * np.transpose(np.matrix(self.y)))
        
        #Compute the mean (each row of K_star adds another element to the array)
        #y_star_mean = np.array( K_star * self.K_inv * self.y[:,np.newaxis] )
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
        #generate the mean for prior distribution(assumed to be zero)
        m = np.transpose( np.matrix( np.zeros(x_star.size) ) )
        #generate the covariance matrix for prior distribution
        K = self.kernel.get_cov_mat(x_star)
        
        return self.get_samples(m,K,n)
    
    def sample_from_posterior(self,x_star,n=1):
        '''
        Draw a desired number of samples from a Gaussian process posterior.
        Based on the given kernel function and the traning data.
        '''
        #get the mean and the covariance matrix of the posterior distribution
        m,K = self.predict(x_star)
        #convert to matrix format
        m = np.transpose( np.matrix(m) )
        
        return self.get_samples(m,K,n)

    def optimize(self):
        '''
        Optimize over the hyperparameters.

        TODO:
        -Add argument to specify optimization method
        -Test this method way more to ensure it performs as expected
        -Ensure that set_hyperparameters updates the kernel hyperparameters as well
        '''
        #Set the initial hyperparameters to the ones entered by the user
        hparams0 = self.kernel.get_hyperparameters()
        #hparams0 = [self.kernel.hparams,self.kernel.hparams,self.kernel.hparams]

        #print(hparams0)
        #print(type(hparams0))
        
        #Minimize the negative log marginal likelihood based on these initial parameters
        res = minimize(self.lml,hparams0,method='Nelder-Mead', tol=1e-6)

        #print(res.x)
        #Update the hyperparameters with the optimal version
        #self.kernel.set_hyperparameters(res.x)

        #Update the covariance matrices for the training data (test matrices updated in predict method)
        self.set_K()
        
        #OPTIONS
        #create new LML class, instantiate a class and pass its bound method to minimize
        #define the LML function inside optimize()
        #result = minimize(self.lml,x0,method='Nelder-Mead', tol=1e-6,args=(self))
        
        return res

    def lml(self,hparams):
        '''
        Negative log marginal likelihood.
        '''

        #print("GFSDGSD")
        
        #Reassign hyperparameters
        self.kernel.set_hyperparameters(hparams)

        #Covariance matrix must be recomputed and inverted every time!
        K = self.kernel.get_cov_mat(self.x)
        K_inv = np.linalg.inv(K)
        
        return np.asscalar( np.matrix(self.y)*K_inv*np.transpose(np.matrix(self.y)) + np.linalg.det(K) )
    
    def get_samples(self,m,K,n=1):
        '''
        General method to sample from a distribution with mean m and covariance matrix K.
        Also accepts the number of samples desired.

        TODO:
        -Fix square root
        -Add positive definite check and loop
        '''
        #identity matrix
        I = np.matrix( np.identity( int(np.sqrt(K.size)) ) )
        
        #add a bit of "noise" because the Cholesky decomposition may not work
        K += 0.0000001*I
        
        #add positive definite check
        #while not positive_definite()
        # K+= i*0.000000000001*I or something like that
        #must have some sort of limit so it isn't adding huge numbers

        #compute the Cholesky decomposition
        L = np.matrix( np.linalg.cholesky(K) )
        
        #generate random samples with mean m and covariance of the identity matrix (i.e., independent)
        u = np.transpose( np.matrix( np.diag( np.random.normal(0,I) ) ) )
        
        #print(u)
        #print(m)
        #print(u+m)
             
        for i in range(1,n,1):
            u = np.append( u,np.transpose(np.matrix(np.diag(np.random.normal(0,I)))),axis=1 )
        
        #print(m)
        #print(L*u)
        
        return m + L*u
