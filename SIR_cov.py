#Run this script to generate inital estimates for robust covariance matrices
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from numpy import log
from SITR import load_globaldata

os.chdir()

#define models
def SIR_model(y, t, beta, gamma, N0):
    S, I, R = y
    dS_dt = -beta*S*I/N0
    dI_dt = beta*S*I/N0 - gamma*I
    dR_dt = gamma*I
    return dS_dt, dI_dt, dR_dt

def SIR_model_ratio(y,t,beta,gamma):
    S, I, R = y
    dS_dt = -beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I
    
    return(dS_dt, dI_dt, dR_dt)


# get a prior for observation noise matrix R

def initial_R(observation):
    obs = observation #confirmed infected people
    obs = N_confirmed

    #induce stationarity
    I1_obs = np.diff(obs)
    I0_obs = np.diff(I1_obs)

    #univariate arma
    def arma_test(observation):
        
        I0_obs = observation

        #check for stationarity
        result = adfuller(I0_obs)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

        arma_mod10 = sm.tsa.ARMA(I0_obs, (1,0)).fit(disp=False)
        print(arma_mod10.params)

        #get residual covariace matrix
        errors = arma_mod10.resid
        R_estimate = np.var(errors)

        return(R_estimate)
    
    R_estimate = arma_test(observation=I0_obs)

    return(R_estimate)


#R_estimate = initial_R(observation = N_confirmed)

def initial_R_ratio(observation):
    obs = observation 
    obs = N_confirmed
    #plt.plot(obs) # behave like exponential process

    #induce stationarity
    I1_obs = np.diff(obs)
    I0_obs = np.diff(I1_obs)
    #plt.plot(I0_obs)
    def relative_growth(obs):
        I1_obs = obs  + 1e-5
        I0_rates = np.diff(I1_obs)/I1_obs[0:-1]
        return(I0_rates)

    I0_rates = relative_growth(obs=I1_obs)


    #univariate arma
    def arma_test(observation):
        
        I0_obs = observation

        #check for stationarity
        result = adfuller(I0_obs)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

        #check for optimal lag order and estimate arma
        arma_mod10 = sm.tsa.ARMA(I0_obs, (1,0)).fit(disp=False)
        print(arma_mod10.params)

        #get residual covariace matrix
        #arma_mod10.__dir__()
        errors = arma_mod10.resid
        R_estimate = np.var(errors)

        return(R_estimate)
    
    #multivariate var
    def var_test():
        pass
    
    R_estimate = arma_test(observation=I0_rates)

    return(R_estimate)




#method 1, get state vector from sample

def build_ensemble(X_b,N):
    N = N # ensemble number
    m,n = np.shape(X_b)
    ensemble = np.zeros(shape=(N,n))
    V_ens = np.zeros(shape=(N,n))

    #create ensemble
    for i, subvector in enumerate(np.split(X_b,N)):
        mu = np.mean(subvector,axis = 0) #mean of each subset
        sigma = np.std(subvector,axis = 0) # st.d of each subset
        ensemble[i] = np.random.normal(mu,sigma) #draw from multivariate normal
        
    ensemble_mean = np.mean(ensemble,axis=0)
    
    #create standard dev matrix nxN
    for i in range(N):

        V_ens[i] = ensemble[i]-ensemble_mean

    B_star = V_ens.T@V_ens

    return(V_ens, B_star)

#run function for empirical X_b
#V_DA,B_DA = build_ensemble(X_b,N=9)


###method2 generate state vector from single ODE

# t=range(200)
def generate_SIR_data(S0,I0,R0,N0,beta0,gamma0,t):

    solve = scipy.integrate.odeint(SIR_model, [S0,I0,R0], t, args =(beta0,gamma0,N0))
    solution = np.array(solve)

    return(solution) #X_b

def generate_SIR_data_ratio(S0,I0,R0,beta0,gamma0,t):

    solve = scipy.integrate.odeint(SIR_model_ratio, [S0,I0,R0], t, args =(beta0,gamma0))
    solution = np.array(solve)

    return(solution) 

###method 3
#generate ensemble from a beta and gamma grid

#gamma_list = np.linspace(0.01,1,10) 
#beta_list = np.linspace(0.01,1.5,15)

def ensemble_cov(SIR_conditions,gamma_list,beta_list,ens_size,timesteps):
    X_b_list = []
    t = timesteps
    N = ens_size
    S0, I0, R0, N0 = SIR_conditions
    for gamma_it in gamma_list:

        for beta_it in beta_list:
            X_b = generate_SIR_data(S0,I0,R0,N0,beta_it,gamma_it,t)
            X_b_list.append(X_b)


    # run all generated X_b series 
    B_list ,V_list = [],[]
    for X_b in X_b_list:
        v, b = build_ensemble(X_b,N=N)
        B_list.append(b)
        V_list.append(v)

        np.shape(B_list)
        #take the mean of 16x3x3 structure to 1x3x3
        B_avg = np.mean(B_list,axis=0)

    return(B_avg)

def ensemble_cov_ratio(SIR_conditions,gamma_list,beta_list,ens_size,timesteps):
    X_b_list = []
    t = timesteps
    N = ens_size
    S0, I0, R0, N0 = SIR_conditions
    for gamma_it in gamma_list:

        for beta_it in beta_list:
            X_b = generate_SIR_data_ratio(S0,I0,R0,beta_it,gamma_it,t)
            X_b_list.append(X_b)


    # run all generated X_b series 
    B_list ,V_list = [],[]
    for X_b in X_b_list:
        v, b = build_ensemble(X_b,N=N)
        B_list.append(b)
        V_list.append(v)

        np.shape(B_list)
        #take the mean of 16x3x3 structure to 1x3x3
        B_avg = np.mean(B_list,axis=0)

    return(B_avg)


#%%
if __name__ == "__main__":
    #Load wuhan data
    os.chdir("...")
    df = pd.read_csv('data/DXY/wuhan_history.csv')
    N_confirmed = df['cum_confirm'].values 

    #set initial conditions
    N0 = 11.08e6
    R0 = N_confirmed[0]
    I0 = 131
    #I0 = 1700

    S0 = N0-I0-R0
    SIR_conditions = S0, I0, R0, N0

    #other countries
    if 1:
        df_global = load_globaldata(country_selector='France')
        df = df_global
        cut = 10
        df = df.iloc[cut:]

        days = np.arange(0, len(df))

        N_confirmed = df['cum_confirm'].values
        df['date'] = df['time']

        days_train, confirmed_train = days, N_confirmed
        days_train = days_train-days_train[0]

        N0 = 66e6 
        R0 = confirmed_train[0]
        I0 = R0*10
        S0 = N0-I0-R0
        S0 = N0-I0-R0
        SIR_conditions = S0, I0, R0, N0





    #estimate R:
    R_prior = initial_R(observation = N_confirmed)
    R_prior_ratio = initial_R_ratio(observation = N_confirmed)

    #prior for Q:
    gamma_list = np.linspace(0.01,1,10) 
    beta_list = np.linspace(0.01,1.5,15)
    ens_size = 20
    
    Q_prior = ensemble_cov(SIR_conditions,gamma_list,beta_list,ens_size,timesteps=range(0,200))

    SIR_conditions_ratio = S0/N0, I0/N0, R0/N0, N0/N0
    Q_prior_ratio = ensemble_cov_ratio(SIR_conditions_ratio,gamma_list,beta_list,ens_size,timesteps=range(0,200))
