import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(12)
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
                 num_visible= 100*100,
                 num_hidden= 100*100, 
                 W= None, 
                 v_bias= None, 
                 h_bias= None,
                 bias= False,
                 T= 1.0,
                 use_cuda= False):
        
        super(RBM, self).__init__()
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
              lr= 0.001, 
              nb_epoch= 1000,
              batch_size= 100, 
              k_learning= 1, 
              test_set= None,
              k_sampling= None,
              FE_dif= None,
              verbose= 1,
              tensorboard= False):
        
        if self.use_cuda:
            training_set = training_set.cuda()
            if test_set is not None:
                test_set = test_set.cuda()
        
        if tensorboard:
            ### TensorBoardX - Initialization
            file_path = 'runs/RBM_nv%d_nh%d_lr%.1E_k%d_bsize%d' % (self.num_visible,
                                                                   self.num_hidden,
                                                                   lr,
                                                                   k_learning,
                                                                   batch_size)
                                                                                 
            self.writer = SummaryWriter(log_dir= file_path)
            W_epoch, v_bias_epoch, h_bias_epoch = self.parameters()
            epoch = 0
            self.writer.add_histogram('W histogram', W_epoch, epoch)
            if self.bias:
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
        #free_en_dif_test_ = [] 
        pseudo_lik_ = []
        t0 = time.time()
        t0_all = time.time()
        ep_list_verb = []
        
        if test_set is not None:
            train_subset = training_set[torch.randperm(test_set.size()[0])]
                     
        for epoch in range(1, nb_epoch + 1):
            
            self.epoch_final = epoch
                  
            train_loss = 0
            test_loss = 0
            s = 0.
            pseudo_lik = 0.
            #free_en_dif = 0.
            #free_en_dif_ = []
            fe_dif_ = []
            
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
                                     
                    # Free energy v0
                    fe = self.free_energy(v0)
                                       
                    # Reconstruction error
                    train_loss += torch.mean( (v0 - vk)**2 ).item()
                    
                    # Pseudo-likelihood
                    rand_flip = (np.arange(v0.shape[0]), 
                                 np.random.randint(0, v0.shape[1], v0.shape[0]))
                    
                    v_ = v0.clone()
                    v_[rand_flip] = 1 - v_[rand_flip]
                    
                    fe_ = self.free_energy(v_)
                    
                    pseudo_lik += v0.shape[1] * torch.log( self.sigmoid( torch.mean(fe_ - fe) ) ).item()
                    
                s += 1.
                
            if epoch%verbose == 0:
                # List storing the quantities which monitor training
                loss_.append(train_loss / s )
                pseudo_lik_.append(pseudo_lik / s)
                # Epoch sequence according to verbose
                ep_list_verb.append(epoch)
                
                # Free energy difference and Reconstruction error in the Test Set
                if test_set is not None:
                    # Reconstruction error in the Test Set
                    vk_test = self.sampling(k_sampling= k_sampling, v0= test_set)
                    test_loss = torch.mean( (test_set - vk_test)**2 ).item()
                    loss_test.append(test_loss)
                    
                    # Free energy difference between test and training set (Hinton)
                    '''
                    After every few epochs, compute the average free energy of a representative subset of 
                    the training data and compare it with the average free energy of a validation set. Always use 
                    the same subset of the training data (from Hinton tutorial)
                    '''
                    fe_valid = self.free_energy(test_set)
                    fe_train = self.free_energy(train_subset)
                    
                    fe_dif = torch.mean(fe_valid).item() - torch.mean(fe_train).item()
                                       
                    fe_dif_.append(fe_dif)
                    
                    if FE_dif:
                        
                        fe_valid = self.free_energy(test_set)
                        fe_train = self.free_energy(train_subset)
                    
                        fe_dif = torch.mean(fe_valid).item() - torch.mean(fe_train).item()
                        fe_dif_.append(fe_dif)
                    
                        t1 = time.time()
                        print('Ep %d, Rec er: %.6f (train), %.6f (test), Pseud_Lik: %.6f, FE_dif: %.6f, Time: %f, ' %(epoch,
                                                                                                                      train_loss/s,
                                                                                                                      test_loss,
                                                                                                                      pseudo_lik / s,
                                                                                                                      fe_dif,
                                                                                                                      t1 - t0))
                    else:
                        t1 = time.time()
                        print('Ep %d, Rec er: %.6f (train), %.6f (test), Pseud_Lik: %.6f, Time: %f, ' %(epoch,
                                                                                                        train_loss/s,
                                                                                                        test_loss,
                                                                                                        pseudo_lik / s, 
                                                                                                        t1 - t0))
                        
                else:                                          
                    t1 = time.time()
                    print('Ep %d, Rec er: %.6f (train), Pseud_Lik: %.6f, Time: %f ' %(epoch,
                                                                                      train_loss/s,
                                                                                      pseudo_lik / s, 
                                                                                      t1 - t0))
                          
                t0 = t1
                
                if tensorboard:
                    W_epoch, v_bias_epoch, h_bias_epoch = self.parameters()
                    self.writer.add_histogram('W histogram', W_epoch, epoch)
                    if self.bias:
                        self.writer.add_histogram('v_bias histogram', v_bias_epoch, epoch)
                        self.writer.add_histogram('h_bias histogram', h_bias_epoch, epoch)
                                                            
        t1_all = time.time()
        print('Total training time = %.3f' % (t1_all - t0_all))
 
        ### Plots after training
        
        # Reconstruction error
        plt.plot(ep_list_verb, loss_, marker= '.', label= 'Training set')
        if test_set is not None:
            plt.plot(ep_list_verb, loss_test, marker= '.', label= 'Test set')
        plt.xlabel('Epoch', fontsize= 15)
        plt.ylabel('Reconstruction Error', fontsize= 15)
        plt.legend(fontsize= 12)
        plt.show()
        
        # Pseudo_Log_likelihood
        plt.plot(ep_list_verb, pseudo_lik_, marker= '.')
        plt.xlabel('Epoch', fontsize= 15)
        plt.ylabel('PseudoLogLikelihood', fontsize= 15)
        plt.show()
        
               
        if test_set is not None:
            
            if FE_dif:
                
                plt.plot(ep_list_verb, free_en_dif_, marker= '.', label= '$FE_{test} - FE_{\train}$')
                plt.xlabel('Epoch', fontsize= 15)
                plt.ylabel('Free Energy Gap', fontsize= 15)
                plt.legend(fontsize= 12)
                plt.show()
       
        
        
        
    def num_train_epochs(self):
        return self.epoch_final
      

    
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
  
    
    
    
    def free_energy(self, v):
        '''
        Function to compute the free energy
        Expression (25) from Hinton's tutorial
        '''
        x = F.linear(v, self.W, self.h_bias)
        va = torch.mv(v, self.v_bias)
                        
        return - va - torch.log(1 + torch.exp(x)).sum(dim=1)
    
    
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))