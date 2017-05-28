import numpy as np
import scipy as sp
from scipy.optimize import minimize

###################
#Gaussian processes
###################

class GP:
    '''
    General class to hold parameters common to both GPR and GPC.
    '''
    def __init__(self,kernel):
        self.kernel = kernel
        self.K = np.array([])
        self.K_inv = np.array([])
        self.x = np.array([])
        self.y = np.array([])

    def compute_K(self,x1,x2):
        '''
        Recompute K, K_inv and return the result.
        '''
        #Compute the covariance matrix and check that it is positive definite.
        K = self.kernel.get_cov_mat(x1,x2)
        K = self.numeric_fix(K)
        
        #Compute the inverse of the covariance matrix.
        K_inv = np.linalg.inv(K)

        return K, K_inv

    def set_K(self):
        '''
        Recompute K, K_inv and set the attributes to the result.
        '''
        self.K, self.K_inv = self.compute_K(self.x,self.x)

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
        I = np.matrix( np.eye(K.shape[0]) )

        #Define the multiple for the identity matrix
        epsilon = 1e-7

        #Define the "rate" at which the multiple should increase.
        alpha = 2
        
        #Define the maximum number of iterations until giving up.
        maxIter = int(1e9)

        #Loop for the maximum number of iterations.
        for i in range(0,maxIter,1):
            if self.positive_definite(K):
                #If the matrix is positive definite, no need to keep adding epsilon*I and can break from loop.
                #print((alpha**i)*epsilon,i)
                break
            #If the matrix is not positive definite, add a small multiple of the identity matrix until it is.
            K += (alpha**i)*epsilon*I
            
        #Return the positive definite covariance matrix.
        return K
        
class GPR(GP):
    '''
    General class for performing Gaussian process regression.
    '''
    def __init__(self,kernel):
        GP.__init__(self,kernel)
        
    def train(self,x,y):
        '''
        Compute and invert the training covariance matrix and store the training data.
        '''
        #Store x and y
        self.x = x
        self.y = y

        #Compute the covariance matrix between the test data and itself
        self.set_K()
        
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

    def optimize(self,method='SLSQP'):
        '''
        Optimize over the hyperparameters.

        TODO:
        -Test this method way more to ensure it performs as expected
        '''
        #Set the initial hyperparameters to the ones entered by the user.
        hparams0 = np.array( self.kernel.get_hyperparameters() )
        hparams_bounds = self.kernel.get_hyperparameters_bounds()
        #print(hparams0)
        
        #Minimize the negative log marginal likelihood based on these initial parameters.
        res = minimize(self.lml,hparams0,method=method, bounds=hparams_bounds ,tol=1e-6)

        #Update the covariance matrices for the training data (test matrices updated in predict method).
        self.set_K()

        #If desired, can return the resulting object from SciPy optimize.
        return res

    def lml(self,hparams=None):
        '''
        Negative log marginal likelihood.
        '''
        #Check to see if an array of hyperparameters is passed.
        if hparams is not None:
            #Reassign hyperparameters if an array of hyperparameters is passed.
            self.kernel.set_hyperparameters(hparams)
            
        #print(hparams)
        
        #Covariance matrix must be recomputed and inverted every time!
        K,K_inv = self.compute_K(self.x,self.x)

        #Return the negative log marginal likelihood (scalar).
        return np.asscalar( np.matrix(self.y)*K_inv*np.transpose(np.matrix(self.y)) + np.linalg.det(K) )
    
    def get_samples(self,m,K,n=1):
        '''
        General method to sample from a distribution with mean m and covariance matrix K.
        Also accepts the number of samples desired.
        Returns n samples drawn from distribution of N~(m,K)
        '''
        #Ensure K is 2D and has same number of rows and columns.
        if (K.ndim != 2) or (K.shape[0] != K.shape[1]):
            errMsg = "The number of rows and columns of this matrix differ ({},{}).".format(K.shape[0],K.shape[1])
            raise TypeError(errMsg)

        #Check that the covariance matrix is positive definite and fix if it is not.
        K = self.numeric_fix(K)
        #print(self.positive_definite(K))
        
        #Compute the Cholesky decomposition.
        L = np.matrix( np.linalg.cholesky(K) )
        
        #Generate random samples with mean **0** and covariance of the identity matrix (i.e., independent).
        v = np.random.normal(0,1,m.size)
        v = v[:,np.newaxis]
        
        #Generate the desired number of samples.
        for i in range(1,n,1):
            tmp = np.random.normal(0,1,m.size)
            tmp = tmp[:,np.newaxis]

            v = np.append( v , tmp , axis=1 )
    
        #Return the sample(s) with distribution N~(m,K).
        return m + L*np.matrix(v)

class GPCB(GP):
    '''
    General class for performing Gaussian process classification (binary case).
    '''
    def __init__(self,kernel):
        GP.__init__(self,kernel)

    def train(self,x,y):
        '''
        Compute and invert the training covariance matrix and store the training data.
        '''
        #Store x and y
        self.x = x
        self.y = y

        #Compute the covariance matrix between the test data and itself
        self.set_K()

    def predict(self,x_star,map_prediction=True):
        #Obtain optimal value of f_hat
        f_hat = np.zeros(self.y.size)
        f_hat = self.newton_method(f_hat,self.y,self.K_inv)
        
        #Compute the covariance matrix between the test data and the training data.
        K_star = self.kernel.get_cov_mat(self.x,x_star)

        #Compute the covariance matrix between the training data and itself.
        K_star_star = self.kernel.get_cov_mat(x_star,x_star)

        #f_hat_tmp = np.reshape( np.array(f_hat),f_hat.size )
        W = -np.matrix( np.diag(self.ddll2(f_hat)) )
        K_prime = self.K + np.linalg.inv(W)
        K_prime_inv = np.linalg.inv(K_prime)

        #Compute the mean (each row of K_star adds another element to the array)
        f_star_mean = np.dot(K_star*self.K_inv,f_hat)
        pi_hat_star_mean = self.pi(f_star_mean) #the sigmoid of the mean of the expectation (MAP prediction)
        
        if map_prediction is True:
            return pi_hat_star_mean

        #Compute the variance by taking the diagonal elements
        f_star_cov = K_star_star - K_star * K_prime_inv * np.transpose(K_star)
        f_star_var = np.diag(f_star_cov)

        f_star_mean = np.reshape(np.array(f_star_mean),f_star_mean.size)
        
        #MacKay approximation of the integral
        pi_star_mean = self.pi(self.kappa(f_star_var)*f_star_mean)
        return pi_star_mean
        
    def kappa(self,f_star_var):
        return np.sqrt( ( 1+np.pi*f_star_var/8 )**(-1) )

    def pi(self,f):
        return 1.0/(1.0+np.exp(-f))
    
    def dll2(self,f,y):
        return (y+1)/2.0 - self.pi(f)

    def ddll2(self,f):
        return -self.pi(f)*(1.0-self.pi(f))

    def f_new(self,f,y,K_inv):
        W = -np.diag(self.ddll2(f))
        #FIX THIS TO RETURN ARRAY PROPERLY!!!
        term1 = np.array( np.linalg.inv( (K_inv+W) ) )
        term2 = np.matmul(W,f) + self.dll2(f,y)
        f_new = np.dot(term1,term2)
        return f_new
        
    def newton_method(self,f_hat_guess,y,K_inv):
        dist = 10000
        f_hat = f_hat_guess
        while(dist>0.001):
            f_hat_new = self.f_new(f_hat,y,K_inv)
            dist = np.linalg.norm(f_hat_new - f_hat)
            f_hat = f_hat_new
            
        return f_hat

class GPC(GP):
    '''
    General class for performing Gaussian process classification (binary case).
    '''
    def __init__(self,kernel):
        GP.__init__(self,kernel)

        self.C = -1
        self.n = -1

        self.x_c = np.array([])
        self.K_c = np.array([])
        self.K_c_inv = np.array([])

    def train(self,x,y,C):
        '''
        Compute and invert the training covariance matrix and store the training data.
        '''
        self.n = int(y.size/C)
        self.C = C
        
        #Store x and y, and also x_c (since x is just a repeat of x_c n times)
        self.x = x
        self.x_c = x[0:self.n]
        self.y = y

        #Compute the covariance matrix between the test data and itself for one class.
        K_c,K_c_inv = self.compute_K(self.x_c,self.x_c)
        self.K_c = K_c
        self.K_c_inv = K_c_inv

        #Compute the covariance matrix between the test data and itself for all classes.
        K = K_c
        for i in range(1,C,1):
            K = sp.linalg.block_diag(K,K_c)
        K_inv = np.linalg.inv(K)
        self.K = K
        self.K_inv = K_inv

    def predict(self,x_star,map_prediction=True):
        #Obtain optimal value of f_hat
        f_hat = np.zeros(self.y.size)
        f_hat = self.newton_method_multi(f_hat,self.y,self.K_inv)
        
        #Compute the covariance matrix between the test data and the training data.
        K_star = self.kernel.get_cov_mat(self.x_c,x_star)

        #Compute the covariance matrix between the training data and itself.
        K_star_star = self.kernel.get_cov_mat(x_star,x_star)

        #Compute the mean (each row of K_star adds another element to the array)
        #FIX THIS!!!
        f_star_mean = K_star*self.K_c_inv*np.transpose(np.matrix(f_hat[0:self.n]))
        #pi_hat_star_mean = self.pi(f_star_mean) #the sigmoid of the mean of the expectation (MAP prediction)

        return f_star_mean
        
    def softmax(self,f):
        f_m = np.reshape(f,(self.C,self.n))
        pi = np.array([])
        
        for i in range(0,f.size,1):
            index = (i)%(self.n)
            f_i = f_m[:,index]
            
            num = np.exp(f[i])
            den = np.sum( np.exp(f_i) )
            
            pi = np.append(pi,num/den)
            
        return pi

    def Pi(self,pi):
        pi = np.reshape(pi,(self.C,self.n))
        
        Pi = np.diag(pi[0,:]) #initialize
        for i in range(1,pi.shape[0],1):
            Pi = np.vstack((Pi,np.diag(pi[i,:])))
            
        return Pi

    def PiPiT(self,pi):
        BigPi = self.Pi(pi)
        return np.dot(BigPi,np.transpose(BigPi))

    def f_new_multi(self,f,y,K_inv):
        pi = self.softmax(f)
        BigPiPiT = self.PiPiT(pi)
        W = np.diag(pi)-BigPiPiT
        
        term1 = np.linalg.inv( (K_inv+W) )
        term2 = np.dot(W,f) + y - pi

        f_new = np.dot(term1,term2)
        return f_new

    def newton_method_multi(self,f_hat_guess,y,K_inv):
        dist = 10000
        f_hat = f_hat_guess
        while(dist>0.001):
            f_hat_new = self.f_new_multi(f_hat,y,K_inv)
            dist = np.linalg.norm(f_hat_new - f_hat)
            f_hat = f_hat_new
            
        return f_hat