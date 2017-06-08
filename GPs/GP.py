import numpy as np
import scipy as sp
from scipy.optimize import minimize
from abc import ABC, abstractmethod

###################
#Gaussian processes
###################

class GP(ABC):
    '''
    General class to hold parameters common to both GPR and GPC.
    '''
    def __init__(self,kernel):
        self.kernel = kernel
        self.K = np.array([])
        self.K_inv = np.array([])
        self.x = np.array([])
        self.y = np.array([])

    def compute_K(self,x1,x2=None):
        '''
        Recompute K, K_inv and return the result.
        '''
        #Compute the covariance matrix and check that it is positive definite.
        if x2 is None:
            K = self.kernel.get_cov_mat(x1)
        else:
            K = self.kernel.get_cov_mat(x1,x2)
            
        K = self.numeric_fix(K)
        
        #Compute the inverse of the covariance matrix.
        K_inv = np.linalg.inv(K)

        return K, K_inv

    def set_K(self):
        '''
        Recompute K, K_inv and set the attributes to the result.
        '''
        self.K, self.K_inv = self.compute_K(self.x)

    def set_K_all(self):
        pass
        
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

    def optimize(self,method='SLSQP'):
        '''
        Optimize over the hyperparameters.
        '''
        #Set the initial hyperparameters to the ones entered by the user.
        hparams0 = np.array( self.kernel.get_hyperparameters() )
        hparams_bounds = self.kernel.get_hyperparameters_bounds()
        #print(hparams0)
        
        #Minimize the negative log marginal likelihood based on these initial parameters.
        res = minimize(self.lml,hparams0,method=method, bounds=hparams_bounds ,tol=1e-6)

        #Update the covariance matrices for the training data (test matrices updated in predict method).
        self.set_K()
        self.set_K_all()
        
        #If desired, can return the resulting object from SciPy optimize.
        return res

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
        L = np.linalg.cholesky(K)
        
        #Generate random samples with mean **0** and covariance of the identity matrix (i.e., independent).
        v = np.random.normal(0,1,m.size)
        v = v[:,np.newaxis]
        
        #Generate the desired number of samples.
        for i in range(1,n,1):
            tmp = np.random.normal(0,1,m.size)
            tmp = tmp[:,np.newaxis]

            v = np.append( v , tmp , axis=1 )
    
        #Return the sample(s) with distribution N~(m,K).
        return m + np.dot(L,v)
        
############################
#Gaussian process regression
############################

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
        K_star_star = self.kernel.get_cov_mat(x_star)
        
        #Compute the mean (each row of K_star adds another element to the array)
        y_star_mean = np.dot( np.dot(K_star,self.K_inv) , self.y )
        
        #Compute the variance by taking the diagonal elements
        y_star_cov = K_star_star - np.dot( np.dot(K_star,self.K_inv) , np.transpose(K_star))
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
        #m = np.transpose( np.matrix( np.zeros(x_star.size) ) )
        m = np.zeros(x_star.size)[:,np.newaxis]
        
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
        #m = np.transpose( np.matrix(m) )
        m = m[:,np.newaxis]
        
        #Return n samples for distribution of N~(m,K)
        return self.get_samples(m,K,n)

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
        K,K_inv = self.compute_K(self.x)

        #Return the negative log marginal likelihood (scalar).
        return np.asscalar( np.matrix(self.y)*K_inv*np.transpose(np.matrix(self.y)) + np.linalg.det(K) )

#################################
#Gaussian process classification
#################################

class GeneralGPC(GP):
    '''
    General class for performing Gaussian process classification. The binary and multiclass case inherit.
    '''
    def __init__(self,kernel):
        GP.__init__(self,kernel)

        self.C = -1
        self.n = -1
        self.x_all = np.array([])
        self.K_all = np.array([])
        self.K_all_inv = np.array([])

    def compute_K_all(self,K):
        if self.C>1:
            K_all = K
            for i in range(1,self.C,1):
                K_all = sp.linalg.block_diag(K_all,K)
            K_all = self.numeric_fix(K_all)
            K_all_inv = np.linalg.inv(K_all)

            return K_all,K_all_inv
        else:
            return np.array([]),np.array([])    

    def set_K_all(self):
        self.K_all,self.K_all_inv = self.compute_K_all(self.K)
        
    def train(self,x,y,C=1):
        '''
        Compute and invert the training covariance matrix and store the training data.
        '''
        self.n = int(y.size/C)
        self.C = C
        
        #Store x and y, and also x_c (since x is just a repeat of x_c n times)
        self.x = x[0:self.n]
        self.x_all = x
        self.y = y

        #Compute the covariance matrix between the test data and itself
        self.set_K()

        #Compute the covariance matrix between the test data and itself for all classes.
        self.set_K_all()

    @abstractmethod
    def f_new(self):
        '''
        This must be implemented in child classes since newton_method depends on it.
        '''
        pass
            
    def newton_method(self,f_hat_guess,y,K_inv,tol=1e-5):
        '''
        Iterate to find f_new and obtain the optimal value f_hat.
        '''
        dist = 10*tol
        f_hat = f_hat_guess
        while(dist>tol):
            f_hat_new = self.f_new(f_hat,y,K_inv)
            dist = np.linalg.norm(f_hat_new - f_hat)
            f_hat = f_hat_new 
        return f_hat

class GPCB(GeneralGPC):
    '''
    General class for performing Gaussian process classification (binary case).
    '''
    def __init__(self,kernel):
        GeneralGPC.__init__(self,kernel)

    def train(self,x,y):
        '''
        If using binary classifier, make it so that there is no option for user to enter the number of classes.
        '''
        GeneralGPC.train(self,x,y)
        
    def predict(self,x_star,map_prediction=True):
        #Obtain optimal value of f_hat
        f_hat = np.zeros(self.y.size)
        f_hat = self.newton_method(f_hat,self.y,self.K_inv)
        
        #Compute the covariance matrix between the test data and the training data.
        K_star = self.kernel.get_cov_mat(self.x,x_star)

        #Compute the covariance matrix between the training data and itself.
        K_star_star = self.kernel.get_cov_mat(x_star)

        #Compute the mean (each row of K_star adds another element to the array)
        f_star_mean = np.dot( np.dot(K_star,self.K_inv) , f_hat )

        #Return the sigmoid of the mean of f_star if desired (MAP prediction).
        if map_prediction is True:
            pi_hat_star_mean = self.pi(f_star_mean)
            return pi_hat_star_mean

        W = -np.diag(self.ddll2(f_hat))
        K_prime = self.K + np.linalg.inv(W)
        K_prime_inv = np.linalg.inv(K_prime)
        
        #Compute the variance by taking the diagonal elements
        f_star_cov = K_star_star - np.dot( np.dot(K_star,K_prime_inv) , np.transpose(K_star) )
        f_star_var = np.diag(f_star_cov)
        
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
        term1 = np.linalg.inv( (K_inv+W) )
        term2 = np.matmul(W,f) + self.dll2(f,y)
        return np.dot(term1,term2)

class GPC(GeneralGPC):
    '''
    General class for performing Gaussian process classification (binary case).
    '''
    def __init__(self,kernel):
        GeneralGPC.__init__(self,kernel)

    def predict(self,x_star,class_number=None):
        n_star = len(x_star)

        #Obtain optimal value of f_hat
        f_hat = np.zeros(self.y.size)
        f_hat = self.newton_method(f_hat,self.y,self.K_all_inv)
        
        #Compute the covariance matrix between the test data and the training data.
        K_star = self.kernel.get_cov_mat(self.x,x_star)

        #Compute the covariance matrix between the training data and itself.
        K_star_star = self.kernel.get_cov_mat(x_star)

        ##########################################################################################
        #Compute the mean (each row of K_star adds another element to the array)
        #index_shift = (class_number-1)*self.n
        #f_star_mean = np.dot( np.dot(K_star,self.K_inv) , f_hat[index_shift:index_shift+self.n] )
        ##########################################################################################

        #Calculate matrices for K_star and K_star_star for all classes (currently only 1 covariance matrix).
        #Note that Q_star here is the transpose of Q_star in Rasmussen.
        Q_star = K_star
        K_star_star_all = K_star_star
        for i in range(1,self.C,1):
            Q_star = sp.linalg.block_diag(Q_star,K_star)
            K_star_star_all = sp.linalg.block_diag(K_star_star_all,K_star_star)

        #Calculate the softmax of the optimal f_hat.
        pi_hat = self.softmax(f_hat)

        #Calculate the matrix W for the optimal value of f_hat.
        W = self.compute_W(pi_hat)
        W = self.numeric_fix(W)

        #Calculate the matrix K_prime for the optimal value of f_hat.
        K_prime = self.K_all + np.linalg.inv(W)
        K_prime = self.numeric_fix(K_prime)
        K_prime_inv = np.linalg.inv(K_prime)

        #Calculate the mean for all classes.
        f_star_mean_all = np.dot(Q_star,self.y-pi_hat)
        
        #Calculate the covariance and variance for all classes.
        f_star_cov = K_star_star_all - np.dot( np.dot(Q_star,K_prime_inv) , np.transpose(Q_star) )
        f_star_var = np.diag(f_star_cov)

        #Estimate pi_star_mean by drawing samples from Gaussian distribution N~(f_star_mean,f_star_cov).
        n_samples = 100
        samples = self.get_samples(f_star_mean_all[:,np.newaxis],f_star_cov,n_samples)

        #print('---')
        
        #Initialize pi_star_mean with the softmax of the first sample.
        pi_star_mean = self.softmax(samples[:,0],n_points=n_star)

        #For each column in samples, softmax and then at the end, average it up.
        for i in range(1,n_samples,1):
            pi_star_mean += self.softmax(samples[:,i],n_points=n_star)
        pi_star_mean /= n_samples

        #If desired, can return the estimate of pi_star for only a given class.
        if class_number is None:
            #return f_star_mean_all
            return pi_star_mean
        else:
            index_shift = (class_number-1)*n_star
            #return f_star_mean_all[index_shift:index_shift+self.n]
            return pi_star_mean[index_shift:index_shift+n_star]
        
    def evaluate(self,pi_star_mean,test_data):
        #test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        #return sum(int(x == y) for (x, y) in test_results)
        n_test = len(test_data)

        pi_star_mean = np.reshape(pi_star_mean,(self.C,n_test))

        #test_results = [(np.argmax(x,y) for x, y in pi_star_mean]

        test_results = []
        for i in range(0,n_test,1):
            res = pi_star_mean[:,i]
            test_results.append( (np.argmax(res), test_data[i][1]) )

        #test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]

        n_correct = sum(int(x == y) for (x, y) in test_results)

        print("{0}/{1}".format(n_correct,len(test_data)))
    
    def softmax(self,f,class_number=None,n_points=None):
        '''
        TODO:
        -optimize
        '''
        if n_points is None:
            n = self.n
        else:
            n = n_points

        #print(n)
            
        f_m = np.reshape(f,(self.C,n))
        pi = np.array([])
        
        for i in range(0,f.size,1):
            index = (i)%(n) #index for the ith training point
            f_i = f_m[:,index] #the column corresponding to that training point (C classes long)
            
            num = np.exp(f[i])
            den = np.sum( np.exp(f_i) )
            
            pi = np.append(pi,num/den)
            
        if class_number is None or class_number<1:
            return pi
        else:
            index_shift = (class_number-1)*n
            return pi[index_shift:index_shift+n]

    def Pi(self,pi):
        pi = np.reshape(pi,(self.C,self.n))
            
        Pi = np.diag(pi[0,:]) #initialize
        for i in range(1,pi.shape[0],1):
            Pi = np.vstack((Pi,np.diag(pi[i,:])))
        return Pi

    def PiPiT(self,pi):
        BigPi = self.Pi(pi)
        return np.dot(BigPi,np.transpose(BigPi))

    def compute_W(self,pi):
        BigPiPiT = self.PiPiT(pi)
        return np.diag(pi)-BigPiPiT
        
    def f_new(self,f,y,K_inv):
        pi = self.softmax(f)
        W = self.compute_W(pi)
        
        term1 = np.linalg.inv( (K_inv+W) )
        term2 = np.dot(W,f) + y - pi

        return np.dot(term1,term2)

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
        K_all,K_all_inv = self.compute_K_all(K)

        f_hat = np.zeros(self.y.size)
        f_hat = self.newton_method(f_hat,self.y,self.K_all_inv)

        pi_hat = self.softmax(f_hat)

        W = self.compute_W(pi_hat)
        W = self.numeric_fix(W)
        
        #Calculate the log marginal likelihood.
        term1 = -0.5*np.dot( np.dot(f_hat,K_all_inv),f_hat ) + np.dot(self.y,f_hat)

        term2 = 0
        f_hat_m = np.reshape(f_hat,(self.C,self.n))
        for i in range(0,self.n,1):
            f_i = f_hat_m[:,i]
            term2 += -np.log( np.sum( np.exp(f_i) ) )

        #W_sqrt = np.sqrt(W)
        #B = np.dot( np.dot(W_sqrt,K_all),W_sqrt)

        #print(W)
        #print(W_sqrt)
        #print(np.dot(W_sqrt,K_all))
        #print(np.diag(W))


        #term3 = -0.5*np.log(np.linalg.det(K_all)*np.linalg.det(K_all_inv+W))

        #np.eye(self.C,self.n)+

        tmpterm1 = np.dot(sp.linalg.sqrtm(W),K_all)
        tmpterm2 = np.dot( tmpterm1 , sp.linalg.sqrtm(W) )

        #print(K_all.shape)
        #print(W.shape)
        #print(sp.linalg.sqrtm(W).shape)
        
        term3 = -0.5*np.log(np.linalg.det(np.eye(self.C*self.n)+tmpterm2))

        #print(term3)

        
        #print(K_all)
        #print(K_all_inv)
        #print(K_inv)
        #print(term1)
        #print(term2)
        #print(K_all_inv+W)
        #print(np.linalg.det(K_all))
        #print(np.linalg.det(K_all_inv+W))
        
        #Return the negative log marginal likelihood (scalar).
        #return np.asscalar( np.matrix(self.y)*K_inv*np.transpose(np.matrix(self.y)) + np.linalg.det(K) )

        return -(term1 + term2 + term3)
