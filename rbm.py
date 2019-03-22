import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import time



class RBM(nn.Module):
    ''' Restricted Boltzmann Machine

    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
        k (integer): k times v->h & h->v sweep in a single contrastive divergence run.
     '''
    
    def __init__(self, 
                 num_visible= 16,
                 num_hidden= 4, 
                 W= None, 
                 v_bias= None, 
                 h_bias= None,
                 bias= False,
                 T= 1.0,
                 use_cuda= False):
        
        super(TRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden  = num_hidden
        self.W = W
        self.v_bias = v_bias
        self.h_bias = h_bias
        self.bias = bias
        self.use_cuda = use_cuda
        self.beta = 1.0 / T
                       
        if self.W is None:

            ### W: weight matrix initialization           
            self.W = nn.init.normal_(torch.empty(self.num_hidden, self.num_visible),
                                     mean= 0,
                                     std= 0.1)       
                
        if self.bias:
                       
            if self.v_bias is None:

                ### v_bias: visible units bias initialization            
                self.v_bias = nn.init.uniform_(torch.empty(self.num_visible),
                                               a= 0,
                                               b =0.001)
                
            if self.h_bias is None:

                ### h_bias: hidden units bias initialization                           
                self.h_bias = nn.init.constant_(torch.empty(self.num_hidden),
                                                val= 0)
                        
        else:
            self.v_bias = nn.init.constant_(torch.empty(self.num_visible),
                                            val= 0)
                                           
            self.h_bias = nn.init.constant_(torch.empty(self.num_hidden),
                                            val= 0)
            
       
        if self.use_cuda:
            self.v_bias = self.v_bias.cuda()
            self.h_bias = self.h_bias.cuda()
            self.W = self.W.cuda()
            
        
        
            
    def v_to_h(self, v):
        '''
        Forward pass: from visible (input) to hidden -> Sample hidden units
        
        Conditional distribution: p(h|v)
        '''
 
        if self.use_cuda:
            v = v.cuda()
                                   
        p_h = torch.sigmoid( self.beta * F.linear(v, self.W, self.h_bias) )
        
        if self.use_cuda:
            p_h = p_h.cuda()
        
        ## Gibbs sampling: hidden given visible
        sample_h = self.sample_from_p(p_h)
        
        return p_h, sample_h

    
    
        
    def h_to_v(self, h):
        '''
        Backward pass: from hidden (input) to visible > Sample visible units
        
        Conditional distribution: p(v|h)
        '''
       
        if self.use_cuda:
            h = h.cuda()
        
        p_v = torch.sigmoid( self.beta * F.linear(h, self.W.t(), self.v_bias) ) 
         
        if self.use_cuda:
            p_v = p_v.cuda()

        ## Gibbs sampling: visible given hidden
        sample_v = self.sample_from_p(p_v)
      
        return p_v, sample_v
    
    
    
    
    def sample_from_p(self,p):
        '''
        Auxiliary function to Gibbs Sampling in forward and backward passes.
        '''
        
        if self.use_cuda:
            p = p.cuda()
            
        rand_prob = torch.rand(p.size())
        
        if self.use_cuda:
            rand_prob = rand_prob.cuda()

        return F.relu( torch.sign( p - rand_prob ) )
    
    
    
    
    def train(self, v0, vk, ph0, phk, lr):
        '''
        Training
        '''
        self.W += (torch.mm(v0.t(), ph0).t() / v0.size()[0] - torch.mm(vk.t(), phk).t() / vk.size()[0] )*lr
        
        if self.bias:
            self.v_bias +=  torch.sum((v0 - vk), 0) *lr / vk.size()[0] 
            self.h_bias +=  torch.sum((ph0 - phk), 0) *lr / vk.size()[0]
    
    
    
    
    def parameters(self):
        '''
        Return RBM parameters
        
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
        '''
        return self.W, self.v_bias, self.h_bias

    
    
    
    def learn(self,
              training_set, 
              lr, 
              nb_epoch,
              batch_size, 
              k_learning, 
              test_set= None,
              k_sampling= None,
              verbose= 1,
              tensorboard= False):
        
        if self.use_cuda:
            training_set = training_set.cuda()
            if test_set is not None:
                test_set = test_set.cuda()
        
        if tensorboard:
            ### TensorBoardX - Initialization
            file_path = 'runs/RBM_25000_nv%d_nh%d_lr%.1E_k%d_bsize%d_epochs%d' % (self.num_visible,
                                                                                  self.num_hidden,
                                                                                  lr,
                                                                                  k_learning,
                                                                                  batch_size,
                                                                                  nb_epoch)
            self.writer = SummaryWriter(log_dir= file_path)
            W_epoch, v_bias_epoch, h_bias_epoch = self.parameters()
            epoch = 0
            self.writer.add_histogram('W histogram', W_epoch, epoch)
            self.writer.add_histogram('v_bias histogram', v_bias_epoch, epoch)
            self.writer.add_histogram('h_bias histogram', h_bias_epoch, epoch)
        
        
        # Print statement just to check the parameters
        print('RBM --- Nv= %d, Nh= %d, lr= %.1E, k= %d, Bsize= %d, Epochs= %d, USE_CUDA= %s, verbose= %d' % (self.num_visible,
                                                                                                             self.num_hidden,
                                                                                                             lr,
                                                                                                             k_learning,
                                                                                                             batch_size,
                                                                                                             nb_epoch,
                                                                                                             self.use_cuda,
                                                                                                             verbose))
        print('Starting training')
        
        loss_ = []
        free_en_dif_ = []
        loss_test = []
        free_en_dif_test_ = []     
        t0 = time.time()
        t0_all = time.time()
                     
        for epoch in range(1, nb_epoch + 1):
                  
            train_loss = 0
            test_loss = 0
            s = 0.
            free_en_dif = 0.
            free_en_dif_test = 0
            
            for j in range(0, training_set.size()[0], batch_size):
                
                vk = training_set[j: j + batch_size]
                v0 = training_set[j: j + batch_size]
                
                ph0,_ = self.v_to_h(v0)
                
                for k in range(k_learning):
                    
                    _, hk = self.v_to_h(vk)
                    _, vk = self.h_to_v(hk)
                    
                phk,_ = self.v_to_h(vk)
                
                # Updating the parameters
                self.train(v0, vk, ph0, phk, lr)
                
                if epoch%verbose == 0:
                                     
                    # Free energy difference within training set 
                    free_energy_v0 = self.free_energy(v0)
                    free_energy_vk = self.free_energy(vk)
                    free_en_dif += torch.mean(free_energy_v0 - free_energy_vk)
                
                    train_loss += torch.mean( (v0 - vk)**2 )
                    
                s += 1.
                
            if epoch%verbose == 0:
                # List storing the quantities which monitor training
                loss_.append((train_loss/s).item())
                free_en_dif_.append((free_en_dif/s).item())
                                      
                # Free energy difference and Reconstruction error in the Test Set
                if test_set is not None:
                    vk_test = self.sampling(k_sampling= k_sampling, v0= test_set)
                    test_loss = torch.mean( (test_set - vk_test)**2 ).item()
                    loss_test.append(test_loss)
                
                    free_energy_v0 = self.free_energy(test_set)
                    free_energy_vk = self.free_energy(vk_test)
                    free_en_dif_test = torch.mean(free_energy_v0 - free_energy_vk).item()               
                    free_en_dif_test_.append(free_en_dif_test)
                 
                # Monitoring training each 10 epochs
                t1 = time.time()
                print('Epoch %d, Rec er: %.6f (train), %.6f (test), FE df: %f (train), %f (test), Time: %f' %(epoch,
                                                                                                              (train_loss/s).item(),
                                                                                                              test_loss,
                                                                                                              (free_en_dif/s).item(),
                                                                                                              free_en_dif_test,
                                                                                                              t1 - t0))            
                t0 = t1
                
                if tensorboard:
                    W_epoch, v_bias_epoch, h_bias_epoch = self.parameters()
                    self.writer.add_histogram('W histogram', W_epoch, epoch)
                    self.writer.add_histogram('v_bias histogram', v_bias_epoch, epoch)
                    self.writer.add_histogram('h_bias histogram', h_bias_epoch, epoch)
                    self.writer.add_scalar('runs/loss_train', (train_loss/s).item(), epoch)
                    self.writer.add_scalar('runs/loss_test', test_loss, epoch)
                    self.writer.add_scalar('runs/fe_train', free_en_dif_test, epoch)
                    self.writer.add_scalar('runs/fe_test', free_en_dif_test, epoch)
                                                            
        t1_all = time.time()
        print('Total training time = %.3f' % (t1_all - t0_all))
 
        # Plots after training to check the evolution of Free Energy Difference and Reconstruction Error
        plt.plot(loss_, label= 'Training set')
        if test_set is not None:
            plt.plot(loss_test, label= 'Test set')
        plt.xlabel('Epoch', fontsize= 15)
        plt.ylabel('Reconstruction Error', fontsize= 15)
        plt.title('$N_h$ = %d' % self.num_hidden, fontsize= 15)
        plt.legend()
        plt.show()
        
        plt.plot(free_en_dif_, label= 'Training set')
        if test_set is not None:
            plt.plot(free_en_dif_test_, label= 'Test set' )
        plt.xlabel('Epoch', fontsize= 15)
        plt.ylabel('Free Energy Difference', fontsize= 15)
        plt.title('$N_h$ = %d' % self.num_hidden, fontsize= 15)
        plt.legend()
        plt.show()
        
      

    
    def sampling(self, k_sampling, v0= None):
                
        if v0 is None:
            v0 = torch.Tensor([2*np.random.randint(0,2) - 1 for i in range(self.num_visible)])
        
        vk = v0.clone()
            
        ph0,_ = self.v_to_h(v0)
                          
        for k in range(k_sampling):
            
            _, hk = self.v_to_h(vk)
            _, vk = self.h_to_v(hk)
            
        phk,_ = self.v_to_h(vk)
                     
        return vk

    
    
    
    def flow(self, n_it_flow, vk= None):
        
        if vk is None:
            vk = torch.Tensor(np.random.randint(0, 2, size= (1000, self.num_visible)))
                         
        RBM_flux = []
        RBM_flux.append(vk)
                          
        for k in range(n_it_flow):
            
            _, hk = self.v_to_h(vk)
            _, vk = self.h_to_v(hk)
            
            RBM_flux.append(vk)

        return RBM_flux    
    
    
    
    
    def flow_last(self, n_it_flow, vk= None):
        # Similar to the method flow, but returns just the last flow     
        if vk is None:
            vk = torch.Tensor(np.random.randint(0, 2, size= (1000, self.num_visible)))
                         
                         
        for k in range(n_it_flow):
            
            _, hk = self.v_to_h(vk)
            _, vk = self.h_to_v(hk)
            
        return vk
    
    
    
    
    def free_energy(self, v):
        '''
        Function to compute the free energy
        Expression (25) from Hinton's tutorial
        '''
        x = F.linear(v, self.W, self.h_bias)
        va = torch.mv(v, self.v_bias)
                        
        return - va - torch.log(1 + torch.exp(x)).sum(dim=1)