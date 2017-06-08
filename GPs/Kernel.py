import numpy as np
from abc import ABC, abstractmethod

################################################################
#General Kernel class from which all kernels should inherit from
################################################################

class Kernel(ABC):
    '''
    Base class for all kernels.
    '''
    def __init__(self):
        self.n_hparam_expect = -1

    def set_n_hparam_expect(self,n_hparam_expect):
        '''
        Set the expected number of hyperparameters.
        All kernels must set this or it will expect -1 hyperparameters which is nonsense!
        '''
        self.n_hparam_expect = n_hparam_expect

    def check_size(self,n_hparam):
        '''
        Check that the number of hyperparameters used is as expected.
        If someone tries to provide too few/too many hyperparameters, it will give an error.
        If a kernel does not set its expected number of hyperparameters, it will give an error.
        '''
        errMsg = "Expected {} hyperparameters but received {}.".format(self.n_hparam_expect,n_hparam)
        if not (n_hparam == self.n_hparam_expect):
            if (self.n_hparam_expect < 0):
                errMsgNeg = " A negative number indicates that the expected number of hyperparameters was not specified in the kernel definition."
                errMsg += errMsgNeg
            raise ValueError(errMsg)
        else:
            return True
        
    @abstractmethod
    def get_size(self):
        '''
        Return the number of hyperparameters in the kernel.
        Each child class (2) must implement.
        '''
        pass

    @abstractmethod
    def get_hyperparameters(self):
        '''
        Return the kernel hyperparameters.
        Each child class (2) must implement.
        '''
        pass

    @abstractmethod
    def get_hyperparameters_bounds(self):
        pass
    
    @abstractmethod
    def set_hyperparameters(self):
        '''
        Set the kernel hyperparameters.
        Each child class (2) must implement.
        '''
        pass
    
    @abstractmethod
    def compute(self,x1,x2):
        '''
        Compute the covariance matrix between two points.
        Must be implemented seperately for each child class of BasicKernel.
        '''
        pass

    def __add__(self,kernel):
        '''
        Summation of self and kernel.
        '''
        return CombinedKernel(self,kernel,"add")

    def __mul__(self,kernel):
        '''
        Multiplication of self and kernel.
        '''
        return CombinedKernel(self,kernel,"mul")
    
    def get_cov_mat(self,x1,x2=None):
        '''
        Takes as inputs two numpy arrays (only accepts 1 feature currently).
        Return kernel result as a numpy matrix.
        Dimension of 1st argument is the number of columns
        Dimension of 2nd argument is the number of rows
        '''        

        #if x2 is None:
        #    x_mn,x_nm = np.meshgrid(x1,x1)
        #else:
        #    x_mn,x_nm = np.meshgrid(x1,x2)

        #return self.compute(x_mn,x_nm)
        #will need to change from size to shape

        #cols = x1.size
        #rows = cols
        #if x2 is None:
        #    cov_mat = np.zeros((rows,cols))
        #else:
        #    rows = x2.size
        #    cov_mat = np.zeros((rows,cols))
        #    
        #for i in range(0,rows,1):
        #    for j in range(0,cols,1):
        #        cov_mat[i,j]=self.compute(x2[i],x1[j])
        #
        #return cov_mat

        #cols = x1.size
        cols = len(x1)
        rows = cols
        if x2 is not None:
            #rows = x2.size
            rows = len(x2)
            cov_mat = np.zeros((rows,cols))
            for i in range(0,rows,1):
                for j in range(0,cols,1):
                    cov_mat[i,j]=self.compute(x2[i],x1[j])
        else:
            #print('The optimal version has been selected.')
            cov_mat = np.zeros((rows,cols))
            for i in range(0,rows,1):
                for j in range(0,i+1,1):
                    #print(i,j)
                    if i==j and i>0:
                        cov_mat[i,j] = cov_mat[0,0]
                    #if i!=j or i==0:
                    else:
                        cov_mat[i,j] = self.compute(x1[i],x1[j])
                        cov_mat[j,i] = cov_mat[i,j]

        #print(cov_mat)
        #print('---------')
                        
        return cov_mat

##################################################################################################
#Subkernel classes BasicKernel and CombinedKernel which represent the two general types of kernels
##################################################################################################
    
class BasicKernel(Kernel):
    '''
    Class to hold basic (i.e., not combined) kernels.
    '''
    def __init__(self,**kwargs):
        #Kernel initializations
        Kernel.__init__(self)

        #BasicKernel initializations
        self.hparams2 = {}
        self.hparams_bounds = {}
        if kwargs is not None:
            for key,value in kwargs.items():
                self.hparams2[key] = value
        self.n = len(self.hparams2)
        
    def get_size(self):
        return self.n
    
    def get_hyperparameters(self):
        return list(self.hparams2.values())

    def get_hyperparameters_bounds(self):
        return tuple(self.hparams_bounds.values())
        
    def set_hyperparameters(self,hparams):
        self.check_size(hparams.size)

        i=0
        for key in self.hparams2.keys():
            self.hparams2[key] = hparams[i]
            i+=1

    def set_hyperparameters_bounds(self,**kwargs):
        if kwargs is not None:
            for key,value in kwargs.items():
                self.hparams_bounds[key] = value
        
    
class CombinedKernel(Kernel):
    '''
    Class to hold the combination of two kernels (e.g., sum, multiplication, etc.).
    '''
    def __init__(self,kernel1,kernel2,operation):
        #Kernel initializations
        Kernel.__init__(self)
        self.set_n_hparam_expect(kernel1.n_hparam_expect + kernel2.n_hparam_expect)

        #CombinedKernel initializations
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        operationList = ["add","mul"]
        if operation not in operationList:
            raise NameError("Kernel operation not found/supported. Please use a valid operation.")
        else:
            self.operation = operation

    def get_size(self):
        return self.kernel1.get_size() + self.kernel2.get_size()

    def get_hyperparameters(self):
        '''
        Returns a list of hyperparameters.
        This method is dependent on the order of the hyperparameters and is dangerous.
        It should only be used for the optimizer, since the order matters for that.
        '''
        return self.kernel1.get_hyperparameters()+self.kernel2.get_hyperparameters()

    def get_hyperparameters_bounds(self):
        '''
        Returns a tuple of hyperparameter bounds.
        This method is dependent on the order of the hyperparameters and bounds and is *very* dangerous.
        It should only be used for the optimizer, since the order matters for that.
        '''
        return self.kernel1.get_hyperparameters_bounds()+self.kernel2.get_hyperparameters_bounds()
    
    def set_hyperparameters(self,hparams):
        '''
        Sets the hyperparameter bounds.
        This method is dependent on the order of the hyperparameters and is dangerous.
        It should only be used for the optimizer, since the order matters for that.
        '''
        self.check_size(hparams.size)
        m = self.kernel1.get_size()
        self.kernel1.set_hyperparameters(hparams[0:m])
        self.kernel2.set_hyperparameters(hparams[m:])
        
    def compute(self,x1,x2):
        if self.operation == "add":
            return self.kernel1.compute(x1,x2) + self.kernel2.compute(x1,x2)
        elif self.operation == "mul":
            return self.kernel1.compute(x1,x2) * self.kernel2.compute(x1,x2)
        else:
            raise NameError("Kernel operation not found/supported. Please use a valid operation.")
    
#############################################################################
#Specific kernel definitions, all of which inherit from the BasicKernel class
#############################################################################
    
class SqExp(BasicKernel):
    '''
    Squared exponential kernel.
    See https://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __init__(self,lengthscale,variance):
        BasicKernel.__init__(self,lengthscale=lengthscale,variance=variance)
        self.set_hyperparameters_bounds(lengthscale_bounds=(1e-5,1e5),variance_bounds=(1e-5,1e5))
        self.set_n_hparam_expect(2)
        
    def compute(self,x1,x2):
        '''
        Return squared exponential result as a numpy array.
        '''
        return self.hparams2['variance']**2 * np.exp( -np.linalg.norm(x1-x2)**2 / (2*self.hparams2['lengthscale']**2) )
        
class RQ(BasicKernel):
    '''
    Rational quadratic kernel.
    See https://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __init__(self,lengthscale,variance,alpha):
        BasicKernel.__init__(self,lengthscale=lengthscale,variance=variance,alpha=alpha)
        self.set_hyperparameters_bounds(lengthscale_bounds=(1e-5,1e5),
                                        variance_bounds=(1e-5,1e5),
                                        alpha_bounds=(1e-5,1e5))
        self.set_n_hparam_expect(3)
        
    def compute(self,x1,x2):
        '''
        Return rational quadratic result as a numpy array.
        '''
        return self.hparams2['variance']**2 * ( 1 + np.linalg.norm(x1-x2)**2 / (2*self.hparams2['alpha']*self.hparams2['lengthscale']**2) )**(-self.hparams2['alpha'])
    
class ExpSine(BasicKernel):
    '''
    Exponential sine squared kernel (Also known as periodic kernel).
    See https://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __init__(self,lengthscale,variance,period):
        BasicKernel.__init__(self,lengthscale=lengthscale,variance=variance,period=period)
        self.set_hyperparameters_bounds(lengthscale_bounds=(1e-5,1e5),
                                        variance_bounds=(1e-5,1e5),
                                        period_bounds=(1e-5,1e5))
        self.set_n_hparam_expect(3)
        
    def compute(self,x1,x2):
        '''
        Return exponential sine (periodic) result as a numpy array.
        '''
        return self.hparams2['variance']**2 * np.exp( - 2*np.sin(np.pi*np.linalg.norm(x1-x2)/self.hparams2['period'])**2 / (self.hparams2['lengthscale']**2) )

class WhiteNoise(BasicKernel):
    '''
    White noise kernel.
    '''
    def __init__(self,noise):
        BasicKernel.__init__(self,noise=noise)
        self.set_hyperparameters_bounds(noise_bounds=(1e-5,1e5))
        self.set_n_hparam_expect(1)
                
    def compute(self,x1,x2):
        '''
        Return the white noise results.
        This method is not great. It assumes that the matrix will be all zeros for K_star.
        As well, if the training and test set happen to be the same shape, this will fail completely.
        '''

        #if x1.shape[0] == x1.shape[1]:
        #    return self.hparams2['noise'] * np.identity(int(np.sqrt(x1.size)))
        #else:
        #    return np.zeros(x1.shape)

        tol = 1e-9
        if np.linalg.norm(x1-x2)<tol:
            return self.hparams2['noise']
        else:
            return 0
