import autograd.numpy.random as npr
import autograd.numpy as np
import autograd.scipy.stats as ss

from models_multivariate_response_surface import transductive



npr.seed(1234)

class feature_dist():

    def __init__(self,mean=0,sd=1):
        self.mean = mean 
        self.sd = sd
    
    def pdf(self,x):
        return(ss.norm.pdf(x,loc=self.mean,scale=self.sd))
    
    def sample(self,num):
        return npr.normal(loc=self.mean,scale=self.sd,size = (num,1))

a1_x = feature_dist(7,2)
a2_x = feature_dist(7,2)

b1_x = feature_dist(11,2)
b2_x = feature_dist(11,2)

x = np.linspace(0, 18,1000)
num_samples = 150

samples_a1 = a1_x.sample(num_samples)
samples_a2 = a2_x.sample(num_samples)
sample_aa = np.squeeze(np.array([samples_a1,samples_a2]))

samples_b1 = b1_x.sample(num_samples)
samples_b2 = b2_x.sample(num_samples)
sample_bb = np.squeeze(np.array([samples_b1,samples_b2]))

def y1_target(x1,x2):
    return np.sin(np.sqrt(x1 ** 2 + x2 ** 2))

def y2_target(x1,x2):
    return np.cos(np.sqrt(x1 ** 2 + x2 ** 2))

def y_target_d2_ind(x1,x2):
    return np.squeeze(np.array([y1_target(x1,x2),y2_target(x1,x2)]))

def y_noise(n,sd):
    return npr.normal(loc=0,scale=sd,size=(n,1))


y_samples_a =  y_target_d2_ind(samples_a1,samples_a2)
y_samples_a1 = np.reshape(y_samples_a[0],(150,1)) + y_noise(num_samples, 0.1)
y_samples_a2 = np.reshape(y_samples_a[1],(150,1)) + y_noise(num_samples, 0.1)

y_samples_b =  y_target_d2_ind(samples_b1,samples_b2)
y_samples_a1 = np.reshape(y_samples_b[0],(150,1)) + y_noise(num_samples, 0.1)
y_samples_b2 = np.reshape(y_samples_b[1],(150,1)) + y_noise(num_samples, 0.1)


model = transductive([2,32,64,2],d_units = 8)
model.train(sample_aa, y_samples_a, sample_bb, 1000)


