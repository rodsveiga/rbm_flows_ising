import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import time


class IsingMC():
            
    def __init__(self, 
                 L= 100, 
                 T_min= 0.5, 
                 T_max= 6.0, 
                 num_T= 26, 
                 MC_eq_sweeps= 200, 
                 MC_states_per_T= 1000, 
                 MC_sweeps_per_state= 100, 
                 include_Tc= True,
                 Tc_MF= True):
        
        ##### 
        T_list_aux = np.linspace(T_max, T_min, num_T)
        
        if T_list_aux[-1] == 0:
            T_list_aux[-1] = 1.0e-6
            
        T_c = 2*2.0
        
        if include_Tc:
            if Tc_MF:
                T_c_fit = 2*2.0
            else:
                T_c_fit =  2 / ( np.log(1 + np.sqrt(2)) )
                
            T_list_aux = np.append(T_list_aux, [T_c_fit - 0.01, T_c_fit, T_c_fit + 0.01])
            T_list_aux = np.sort(T_list_aux)[::-1]
        
        print('Method fit will construct states to the following temperatures: ')
        print(T_list_aux)
 
        ##### ISING MODEL PARAMETERS
        self.T_list = T_list_aux
        self.L = L
        self.N_spins = L*L
        self.spins = np.array([2*np.random.randint(0,2) - 1 for i in range(L*L)])
        self.T_c = T_c
        
        ##### MONTE CARLO PARAMETERS
        # Number of equilibration sweeps
        self.n_eqSweeps = MC_eq_sweeps  
        # Total number of states per temperature
        self.n_states = MC_states_per_T
        # Number of sweeps performed in on spin state
        self.n_sweeps_per_state = MC_sweeps_per_state
        
       
    ##### MONTE CARLO PROCEDURE    
    def fit(self, thermodynamics= False):
        
        if thermodynamics:
            self.spin_MC_df = []
        
            for self.T in self.T_list:
                print('\nT = %f' %self.T)
                t0 = time.time()
                
                self.m_CW = self.CW_equation()
            
                # Equilibrium thermalization sweeps
                for i in range(self.n_eqSweeps):
                    self.sweep()
                
                # Generating more states for measurements
                for i in range(self.n_states):
                    for j in range(self.n_sweeps_per_state):
                        self.sweep()
                    
                    # Randomly multiply by +1 or -1 to ensure generation of configurations
                    # with both positive and negative magnetization
                    random_flip = 2*np.random.randint(0,2) - 1 
                        
                    self.spin_MC_df.append({'state': np.array( ((random_flip*self.spins) + 1)/2, dtype= int),
                                            'magn': random_flip * self.getMag(),
                                            'energy': self.getEnergy(),'temp': self.T}) 
             
                t1 = time.time()
                print('time = %.2f s' % (t1-t0))
                
        else:
            self.spin_MC_df = []
        
            for self.T in self.T_list:
                print('\nT = %f' %self.T)
                t0 = time.time()
                
                self.m_CW = self.CW_equation(self)
            
                # Equilibrium thermalization sweeps
                for i in range(self.n_eqSweeps):
                    self.sweep()
                
                # Generating more states for measurements
                for i in range(self.n_states):
                    for j in range(self.n_sweeps_per_state):
                        self.sweep()
                    
                    # Randomly multiply by +1 or -1 to ensure generation of configurations
                    # with both positive and negative magnetization
                    random_flip = 2*np.random.randint(0,2) - 1 
                    
                    self.spin_MC_df.append({'state': np.array( ((random_flip*self.spins) + 1)/2, dtype= int), 
                                            'temp': self.T})
                              
            t1 = time.time()
            print('time = %.2f s' % (t1-t0))
                            
        return self
                           
    ##### AUXILIARY FUNCTION: NEAREST NEIGHBORS    
    def neighbours(self):
        
        neighbours = np.zeros((self.N_spins,4), dtype= np.int)
        
        ##### Store each spin's four nearest neighbours in a neighbours array (using periodic boundary conditions)
        for i in range(self.N_spins):
            #### Neighbour to the 'right'
            neighbours[i,0] = i + 1
            # Periodic boundary conditions
            if i%self.L == (self.L-1):
                neighbours[i,0] = i + 1 - self.L
        
            #### 'Upwards' neighbour
            neighbours[i,1] = i + self.L
            # Periodic boundary conditions
            if i >= (self.N_spins-self.L):
                neighbours[i,1] = i + self.L - self.N_spins
        
            #### Neighbour to the 'left'
            neighbours[i,2] = i - 1
            # Periodic boundary conditions
            if i%self.L == 0:
                neighbours[i,2] = i - 1 + self.L
        
            #### 'Downwards' neighbour
            neighbours[i,3] = i - self.L
            # Periodic boundary conditions
            if i <= (self.L-1):
                neighbours[i,3] = i - self.L + self.N_spins
                
        return neighbours
    
    def CW_equation(self):
        
        beta = 1./self.T
        
        if self.T > 4.0:
            x0 = 0.001
        else:
            x0 = 0.999
            
        res = minimize(lambda x: (x - np.tanh(4*beta*x))**2, 
                       x0= x0)
        
        return res.x[0]        
          
    ##### AUXILIARY FUNCTION: TOTAL ENERGY OF EACH STATE 
    def getEnergy(self):
        
        m = self.m_CW  
                       
        Energy = - 2*2*m*np.sum(self.spins) + self.N_spins*2*m*m        
        
        return Energy
    
    ##### AUXILIARY FUNCTION: TOTAL MAGNETIZATION OF EACH STATE
    def getMag(self):
        return np.sum(self.spins)
    
    ##### AUXILIARY FUNCTION: MONTE CARLO SWEEP
    def sweep(self):
        
        site = np.random.randint(0, self.N_spins)
        
        deltaE = 4*2*self.m_CW*self.spins[site]
                 
        if (deltaE <= 0) or (np.random.random() < np.exp(- deltaE / self.T) ):
            # Flip the spin
            self.spins[site] = -self.spins[site]
                
    ##### THERMODYNAMIC QUANTITIES            
    def plot_thermodynamics(self, spin_MC= None, Tc_scaled= False):
        
        if spin_MC is None:
            spin_MC = pd.DataFrame(self.spin_MC_df)
        
        if Tc_scaled:
            spin_MC['temp'] = spin_MC['temp'] / self.T_c
               
        expec_val_per_spin_ENERGY = spin_MC['energy'].groupby(spin_MC['temp']).mean() / self.N_spins
        
        expec_val_per_spin_MAGN = abs(spin_MC['magn']).groupby(spin_MC['temp']).mean() / self.N_spins
                   
        expec_val_ENERGY_square = (spin_MC['energy']**2).groupby(spin_MC['temp']).mean()
        temp_array = np.array(expec_val_ENERGY_square.index, dtype = np.float64)
        expec_val_ENERGY = self.N_spins*expec_val_per_spin_ENERGY
        specific_heat_per_spin = (expec_val_ENERGY_square  - expec_val_ENERGY**2) / (self.N_spins * (temp_array**2))
        
        expec_val_MAGN_square = (spin_MC['magn']**2).groupby(spin_MC['temp']).mean()
        expec_val_MAGN  = self.N_spins*expec_val_per_spin_MAGN 
        mag_suscep_per_spin = (expec_val_MAGN_square  - expec_val_MAGN**2) / (self.N_spins * temp_array)
        
        plt.figure(figsize=(12,9))

        #####
        plt.subplot(221)
        plt.plot(expec_val_per_spin_ENERGY, 'o-')
        
        if Tc_scaled:
            plt.axvline(x= 1.0, color='k', linestyle='--')
            plt.xlabel('$T / T_c$', fontsize=14)
        else:
            plt.axvline(x= self.T_c, color='k', linestyle='--')
            plt.xlabel('$T$', fontsize=14)
                       
        plt.ylabel('$<E>/N$', fontsize=14)

        #####
        
        plt.subplot(222)
        plt.plot(expec_val_per_spin_MAGN, 'o-')
        if Tc_scaled:
            plt.axvline(x= 1.0, color='k', linestyle='--')
            plt.xlabel('$T / T_c$', fontsize=14)
        else:
            plt.axvline(x= self.T_c, color='k', linestyle='--') 
            plt.xlabel('$T$', fontsize=14)
        
        plt.ylabel('$<M>/N$', fontsize=14)

        #####
        
        plt.subplot(223)
        plt.plot(specific_heat_per_spin, 'o-')
        if Tc_scaled:
            plt.axvline(x= 1.0, color='k', linestyle='--')
            plt.xlabel('$T / T_c$', fontsize=14)   
        else:
            plt.axvline(x= self.T_c, color='k', linestyle='--')
            plt.xlabel('$T$', fontsize=14)
       
        plt.ylabel('$<C>/N$', fontsize=14)

        #####
        
        plt.subplot(224)
        plt.plot(mag_suscep_per_spin, 'o-')
        if Tc_scaled:
            plt.axvline(x= 1.0, color='k', linestyle='--')
            plt.xlabel('$T / T_c$', fontsize=14)
        else:
            plt.axvline(x= self.T_c, color='k', linestyle='--') 
            plt.xlabel('$T$', fontsize=14)
            
        plt.ylabel('$< \chi>/N$', fontsize=14)
        
        #####

        plt.suptitle('%d x %d Ising model - Mean Field' %(self.L, self.L), fontsize=18)
        plt.show()
        
    #### THERMODYNAMICS MONTE CARLO DATAFRAME    
    def thermodymanics_MC_df(self):
        return pd.DataFrame(self.spin_MC_df)
        
    ##### DATASET    
    def data(self):
        data = pd.DataFrame(self.spin_MC_df, columns=['state', 'temp'])
        data['phase'] = ( (data['temp'] - self.T_c) / abs(data['temp'] - self.T_c) ).astype(int)
        
        data = pd.get_dummies(data, columns=['phase'])
        data = data.rename(index=str, columns={"phase_-1": "ordered", "phase_1": "desordered"})
        
        return data
    
    
    
    #### THERMODYNAMICS MONTE CARLO DATAFRAME  
          
    def correlations(self, 
                     data= None, 
                     neig= None, 
                     plot_SiSj= False,
                     correlation_length= False):
        
        if neig is None:
            neig = int(self.L / 2)
            
        if neig > int(self.L / 2):
            neig = int(self.L / 2)
            
        if data is None:
            data = pd.DataFrame(self.spin_MC_df, columns=['state', 'temp'])
            
        CORR = []
        CORR_red = []
        NN = self.neighbours()
        
        print('Pairwise correlation calculation')
        
        for N_neig in range(neig):
            
            t0 = time.time()
            print('\nneighbour: %d' % (N_neig + 1))
                            
            corr_ = []
            corr_red = []
            temp_ = []
                           
            for _, temp in enumerate(sorted(data['temp'].value_counts().index.tolist())):
                       
                states_temp = data.loc[data['temp'] == temp]['state']
                corr_state = []
                mag_state = []
                 
                for _, n in enumerate(states_temp.index.tolist()):
                    
                    state_n = np.sign(states_temp[n]- 0.5)       
                    SiSj = 0.
                    
                    for i in range(0, self.N_spins, self.L):
                        
                        for j in range(i, i + self.L):
                            
                            nnn0 = NN[j][0] + N_neig
                            nnn1 = NN[j][1] + N_neig*self.L
                            
                            if nnn0 >= (i + self.L):
                                nnn0 = nnn0 - self.L
                                
                            if nnn1 >= self.N_spins:
                                nnn1 = nnn1 - self.N_spins
                                
                
       
                    SiSj += state_n[j] * state_n[nnn0] + state_n[j] * state_n[nnn1]
                    
                    corr_state.append(SiSj / (2.0*self.N_spins)  )
                    mag_state.append(np.mean(state_n) * np.mean(state_n) )
        
        
                corr_.append(np.mean(corr_state))
                corr_red.append(np.mean(corr_state) - np.mean(mag_state)  )
                
                temp_.append(temp)
        
            CORR.append(corr_)
            CORR_red.append(corr_red)
            
            t1 = time.time()
            print('time= %.3f s' % (t1-t0))
            
        self.temp_ = temp_
        self.CORR = CORR
        self.CORR_red = CORR_red
            
        # Plot <S(r)S(0)> just for checking
        
        if plot_SiSj:
            for k in range(0, len(temp_), 2):
                x = []
                y = []
                
                for j in range(neig):
                    x.append(j+1)
                    y.append(CORR[j][k])
        
                plt.plot(x, y, 'o-', label= 'T = %f' % temp_[k])

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('$r$', fontsize= 20)
            plt.ylabel('$ < S(0) S(r) > $', fontsize= 15)
            plt.xscale('log')
            plt.show()
            
        
        if plot_SiSj:
            for k in range(0, len(temp_), 2):
                x = []
                y = []
                
                for j in range(neig):
                    x.append(j+1)
                    y.append(CORR_red[j][k])
        
                plt.plot(x, y, 'o-', label= 'T = %f' % temp_[k])

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('$r$', fontsize= 20)
            plt.ylabel('$ < S(0) S(r) > -  < S(0) > <S(r) > $', fontsize= 15)
            plt.xscale('log')
            plt.show()
            
            
        # Fitting the correlation length
        
        if correlation_length:
            
                   
            print('\nFitting correlation versus distance to get the correlation length as a function of temperature')
            
            def F(x, A, B, C, D):
                return A * np.exp(-x / B) / (x**C) + D
            
            bounds = ((-np.inf, 0, 0, -np.inf), (np.inf, np.inf, 1, np.inf))
        
            corr_length = []
        
        
            for k in range(0, len(temp_)):
                
                x = []
                y = []

                for j in range(0, neig):
                    x.append(j+1)
                    y.append(CORR[j][k])
                
                popt, pcov = curve_fit(F, x, y, bounds= bounds)
                corr_length.append(popt[1])
            
            # Plotting the correlation length
        
            plt.plot(temp_ / self.T_c, corr_length, 'o-')
            plt.ylabel('$\\xi$', fontsize= 20)
            plt.xlabel('$T / T_c$', fontsize= 20)
            plt.axvline(x= 1.0, color='k', linestyle=':')
            plt.show()
        
            self.corr_length = corr_length   
        
        
    def get_correlation_length(self):
        return self.temp_, self.corr_length
    
    def get_correlations(self, truncated= False):
        
        if truncated:
            corr_out = self.CORR_red
        else:
            corr_out = self.CORR
            
        return self.temp_, corr_out
