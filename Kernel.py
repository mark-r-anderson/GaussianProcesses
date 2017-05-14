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

    def check_size(self):
        '''
        NOT ACTUALLY USED ANYWHERE YET!!!
        NEED TO HAVE INPUT FOR DIMENSION OF hparam; DEFAULT TO n=self.get_size()

        Check that the number of hyperparameters used is as expected.
        If someone tries to provide too few/too many hyperparameters, it will give an error.
        If a kernel does not set its expected number of hyperparameters, it will give an error.
        '''
        if not (self.get_size() == self.n_hparam_expect):
            raise ValueError("Hyperparameter array does not match expected size of hyperparameters.")
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
    
    TODO:
    -add check in __int__ for *args
    '''
    def __init__(self,*args):
        #Kernel initializations
        Kernel.__init__(self)

        #BasicKernel initializations
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
        self.check_size()
            
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
        Retrieves the hyperparameters.
        Each list and sublist respresents a kernel.
        The innermost list will contain the arrays for the two basic kernels.
        '''
        #return [self.kernel1.get_hyperparameters(),self.kernel2.get_hyperparameters()]
        #return np.array([self.kernel1.get_hyperparameters(),self.kernel2.get_hyperparameters()],dtype=object)

        return np.append(self.kernel1.get_hyperparameters(),self.kernel2.get_hyperparameters())
    
    def set_hyperparameters(self,hparams):
        '''
        Set the hyperparameters.

        TODO:
        -check size of hparams, throw error
        -option to use organizedList
        '''
        #self.kernel1.set_hyperparameters(hparams[0])
        #self.kernel2.set_hyperparameters(hparams[1])

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
        #BasicKernel initializations
        BasicKernel.__init__(self,lengthscale,variance)
        self.set_n_hparam_expect(2)

        #SqExp initializations
        self.lengthscale = lengthscale
        self.variance = variance

    def set_hyperparameters(self,hparams):
        '''
        TODO:
        -remove!
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
    See https://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __init__(self,lengthscale,variance,alpha):
        #BasicKernel initializations
        BasicKernel.__init__(self,lengthscale,variance,alpha)

        #RQ initializations
        self.lengthscale = lengthscale
        self.variance = variance
        self.alpha = alpha

    def set_hyperparameters(self,hparams):
        '''
        TODO:
        -remove!
        '''
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
    See https://www.cs.toronto.edu/~duvenaud/cookbook/index.html for details.
    '''
    def __int__(self,lengthscale,variance,period):
        #BasicKernel initializations
        BasicKernel.__init__(self)

        #ExpSine initializations
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period

    def set_hyperparameters(self,hparams):
        '''
        TODO:
        -remove!
        '''
        self.lengthscale = hparams[0]
        self.variance = hparams[1]
        self.period = hparams[2]
        
    def compute(self,x1,x2):
        '''
        Return exponential sine (periodic) result as a numpy array.
        '''
        
        return self.variance**2 * np.exp( - 2*np.sin(np.pi*abs(x1-x2)/self.period) / (self.lengthscale**2) )
