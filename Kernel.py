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
        pass
    
    @abstractmethod
    def get_size(self):
        '''
        Return the number of hyperparameters in the kernel.
        '''
        pass
        
    @abstractmethod
    def set_hyperparameters(self):
        '''
        Set the kernel hyperparameters.
        '''
        pass

    @abstractmethod
    def get_hyperparameters(self):
        '''
        Return the kernel hyperparameters.
        '''
        pass
    
    @abstractmethod
    def compute(self,x1,x2):
        '''
        Compute the covariance matrix between two points.
        Must be implemented seperately for each kernel as every kernel has a different equation.
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
        
        TODO:
        -optimize
        '''        
        if x2 == None:
            x_mn,x_nm = np.meshgrid(x1,x1)
        else:
            x_mn,x_nm = np.meshgrid(x1,x2)
    
        return np.matrix( self.compute(x_mn,x_nm) )

##################################################################################################
#Subkernel classes BasicKernel and CombinedKernel which represent the two general types of kernels
##################################################################################################
    
class BasicKernel(Kernel):
    '''
    Class to hold basic (i.e., not combined) kernels.
    '''
    def __init__(self,*args):
        Kernel.__init__(self)
        self.hparams = np.array([])
        for arg in args:
            self.hparams = np.append(self.hparams,arg)
        self.n = self.hparams.size
            
    def get_size(self):
        return self.n
            
    def get_hyperparameters(self):
        return self.hparams

    def set_hyperparameters(self,hparams):
        self.hparams = hparams
            
class CombinedKernel(Kernel):
    '''
    Class to hold the combination of two kernels (e.g., sum, multiplication, etc.).
    '''
    def __init__(self,kernel1,kernel2,operation):
        Kernel.__init__(self)
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.operation = operation

    def compute(self,x1,x2):
        if self.operation == "add":
            return self.kernel1.compute(x1,x2) + self.kernel2.compute(x1,x2)
        if self.operation == "mul":
            return self.kernel1.compute(x1,x2) * self.kernel2.compute(x1,x2)

    def get_size(self):
        return self.kernel1.get_size() + self.kernel2.get_size()
        
    def set_hyperparameters(self,hparams):
        '''
        Set the hyperparameters.

        TODO:
        -check size of hparams, throw error
        '''
        #self.kernel1.set_hyperparameters(hparams[0])
        #self.kernel2.set_hyperparameters(hparams[1])

        m = self.kernel1.get_size()
        self.kernel1.set_hyperparameters(hparams[0:m])
        self.kernel2.set_hyperparameters(hparams[m:])
        
    def get_hyperparameters(self):
        '''
        Retrieves the hyperparameters.
        Each list and sublist respresents a kernel.
        The innermost list will contain the arrays for the two basic kernels.
        '''
        #return [self.kernel1.get_hyperparameters(),self.kernel2.get_hyperparameters()]
        #return np.array([self.kernel1.get_hyperparameters(),self.kernel2.get_hyperparameters()],dtype=object)

        return np.append(self.kernel1.get_hyperparameters(),self.kernel2.get_hyperparameters())
        
#############################################################################
#Specific kernel definitions, all of which inherit from the BasicKernel class
#############################################################################
    
class SqExp(BasicKernel):
    '''
    Squared exponential kernel.
    See http://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __init__(self,lengthscale,variance):
        BasicKernel.__init__(self,lengthscale,variance)
        self.lengthscale = lengthscale
        self.variance = variance

    def set_hyperparameters(self,hparams):
        '''
        Set the hyperparameters.

        TODO:
        -implement a warning message if the number of elements in the array differs from expectation
        -implement the warning message as a general function in base class Kernel
        '''
        BasicKernel.set_hyperparameters(self,hparams)
        self.lengthscale = hparams[0]
        self.variance = hparams[1]
        
    def compute(self,x1,x2):
        '''
        Return squared exponential result as a numpy array.
        '''
        
        return self.variance**2 * np.exp( -(x1-x2)**2 / (2*self.lengthscale**2) )
        
class RQ(BasicKernel):
    '''
    Rational quadratic kernel.
    See http://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __init__(self,lengthscale,variance,alpha):
        BasicKernel.__init__(self,lengthscale,variance,alpha)
        self.lengthscale = lengthscale
        self.variance = variance
        self.alpha = alpha

    def set_hyperparameters(self,hparams):
        BasicKernel.set_hyperparameters(self,hparams)
        self.lengthscale = hparams[0]
        self.variance = hparams[1]
        self.alpha = hparams[2]
        
    def compute(self,x1,x2):
        '''
        Return rational quadratic result as a numpy array.
        '''
        
        return self.variance**2 * ( 1 + np.exp( -(x1-x2)**2 / (2*self.alpha*self.lengthscale**2) ) )
    
class ExpSine(BasicKernel):
    '''
    Exponential sine squared kernel (Also known as periodic kernel).
    See http://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __int__(self,lengthscale,variance,period):
        BasicKernel.__init__(self)
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period

    def set_hyperparameters(self,hparams):
        self.lengthscale = hparams[0]
        self.variance = hparams[1]
        self.period = hparams[2]
        
    def compute(self,x1,x2):
        '''
        Return exponential sine (periodic) result as a numpy array.
        '''
        
        return self.variance**2 * np.exp( - 2*np.sin(np.pi*abs(x1-x2)/self.period) / (self.lengthscale**2) )
