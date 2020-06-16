"""
Run this code to generate results for the coupled network architecture
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from numpy import log
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from tqdm import tqdm

#change to correct filepath
os.chdir("...")

#%%
#Run Cov Estimation
def load_globaldata(country_selector = 'United Kingdom'):

    df_conf = pd.read_csv('.../time_series_covid_19_confirmed.csv')
    df_rec = pd.read_csv('.../time_series_covid_19_recovered.csv')
    df_dec = pd.read_csv('.../time_series_covid_19_deaths.csv')

    def extract_country(df_raw):
        
        df_conf = df_raw
        #select countries
        df_countries = df_conf[df_conf['Country/Region'].str.match('US|Italy|New York State|Belgium|Sweden|Germany|United Kingdom|France|Korea|Japan|Spain|Cambodia|Pakistan|India|Brazil|Philippines|South Africa|Nigeria|Turkey')]

        #select only national level
        df_countries['Province/State'] = df_countries['Province/State'].fillna(df_conf['Country/Region'])
        df_countries = df_countries[df_countries['Province/State'].str.match('US|Italy|New York State|Belgium|Sweden|Germany|United Kingdom|France|Korea|Japan|Spain|Cambodia|Pakistan|India|Brazil|Philippines|South Africa|Nigeria|Turkey')]

        #transpose
        df_countries = df_countries.drop(['Province/State','Lat', 'Long'],axis=1)
        df_countries = df_countries.transpose()

        #reset indeces
        df_countries.columns = df_countries.loc['Country/Region',:]
        df_countries = df_countries.drop(['Country/Region'])

        df_countries.plot()

        return(df_countries)
    
    df_conf = extract_country(df_conf)
    df_rec = extract_country(df_rec)
    df_dec = extract_country(df_dec)

    #inital Wuhan Data estimate
    if 1:
        df_wuhan = df # optional: select wuhan dataset
        df_global = df_wuhan.reindex(range(0,len(df_conf)))

        df_global.index = df_conf.index
        df_global['time'] = df_conf.index
        df_global['cum_confirm'] = df_conf[country_selector]
        df_global['cum_heal'] = df_rec[country_selector]
        df_global['cum_dead'] = df_dec[country_selector]
        df_global.plot()
        print(country_selector)
        print('Printing confirmed cases for:', df_global['cum_confirm'].tail())

    return(df_global)

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
    obs = observation #in our case infected people
    obs = N_confirmed
    #plt.plot(obs) # behave like exponential process

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

        #check for optimal lag order and estimate arma
        arma_mod10 = sm.tsa.ARMA(I0_obs, (1,0)).fit(disp=False)
        print(arma_mod10.params)

        #get residual covariace matrix
        errors = arma_mod10.resid
        R_estimate = np.var(errors)

        return(R_estimate)    
    #multivariate var
    def var_test():
        pass
    
    R_estimate = arma_test(observation=I0_obs)

    return(R_estimate)

def initial_R_ratio(observation):
    obs = observation #in our case infected people
    obs = N_confirmed

    #induce stationarity
    I1_obs = np.diff(obs)
    I0_obs = np.diff(I1_obs)

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
        errors = arma_mod10.resid
        R_estimate = np.var(errors)

        return(R_estimate)
    
    #multivariate var
    def var_test():
        pass
    
    R_estimate = arma_test(observation=I0_rates)

    return(R_estimate)

# #create state vector
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
#generate state vector from single ODE
t=range(200)
def generate_SIR_data(S0,I0,R0,N0,beta0,gamma0,t):

    solve = scipy.integrate.odeint(SIR_model, [S0,I0,R0], t, args =(beta0,gamma0,N0))
    solution = np.array(solve)

    return(solution) #X_b

def generate_SIR_data_ratio(S0,I0,R0,beta0,gamma0,t):

    solve = scipy.integrate.odeint(SIR_model_ratio, [S0,I0,R0], t, args =(beta0,gamma0))
    solution = np.array(solve)

    return(solution) 

#generate ensemble from a beta and gamma grid
gamma_list = np.linspace(0.01,1,10) # unlikely that gamma is above 1, would mean infected get removed in less than  a day
beta_list = np.linspace(0.01,1.5,15)

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

#Optional: Load wuhan data
if 1:
    #os.chdir("...")
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
    #'Cambodia|Pakistan|India|Brazil|Philippines|South Africa|Nigeria|Turkey'
    country_population = {'US':328e6,'New York State': 19e6,'Japan': 126e6,'Korea, South':51e6,'Sweden':10e6,'Italy': 60e6,'Germany': 82e6,'Belgium':11e6,'United Kingdom':66e6,'France':66e6,'Cambodia': 16e6, 'Pakistan': 197e6,'India':1339e6,'Brazil':209e6,'Philippines':104e6,'South Africa': 56e6,'Nigeria':190e6,'Turkey':80e6}
    country_selector = 'Brazil'
    df_global = load_globaldata(country_selector)
    df = df_global
    cut = 55 #in-sample size
    hold_out_test = 5
    df = df.iloc[cut:-hold_out_test]

    days = np.arange(0, len(df))

    N_confirmed = df['cum_confirm'].values #alternative
    df['date'] = df['time']

    days_train, confirmed_train = days, N_confirmed
    days_train = days_train-days_train[0]

    #N0 = 60e6
    N0 = country_population[country_selector] 
    R0 = confirmed_train[0]
    I0 = R0*10 #Initial latent infection cases
    S0 = N0-I0-R0
    S0 = N0-I0-R0
    SIR_conditions = S0, I0, R0, N0

#Run Covariance Functions
#estimate R:
R_prior = initial_R(observation = N_confirmed)
R_prior_ratio = initial_R_ratio(observation = N_confirmed)

#prior for Q:
gamma_list = np.linspace(0.01,1,10) # unlikely that gamma is above 1, would mean infected get removed in less than  a day
beta_list = np.linspace(0.01,1.5,15)
ens_size = 20

Q_prior = ensemble_cov(SIR_conditions,gamma_list,beta_list,ens_size,timesteps=range(0,200))

SIR_conditions_ratio = S0/N0, I0/N0, R0/N0, N0/N0
Q_prior_ratio = ensemble_cov_ratio(SIR_conditions_ratio,gamma_list,beta_list,ens_size,timesteps=range(0,200))

#%%
#Run Country Configuration

#Optional: Load wuhan data
#os.chdir("...")
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
    #'Cambodia|Pakistan|India|Brazil|Philippines|South Africa|Nigeria|Turkey'
    country_population = {'US':328e6,'New York State': 19e6,'Japan': 126e6,'Korea, South':51e6,'Sweden':10e6,'Italy': 60e6,'Germany': 82e6,'Belgium':11e6,'United Kingdom':66e6,'France':66e6,'Cambodia': 16e6, 'Pakistan': 197e6,'India':1339e6,'Brazil':209e6,'Philippines':104e6,'South Africa': 56e6,'Nigeria':190e6,'Turkey':80e6}
    country_selector = 'Brazil'
    df_global = load_globaldata(country_selector)
    df = df_global
    cut = 10  #20
    hold_out_test = 5
    df = df.iloc[cut:-hold_out_test]

    days = np.arange(0, len(df))

    N_confirmed = df['cum_confirm'].values #alternative
    df['date'] = df['time']

    days_train, confirmed_train = days, N_confirmed
    days_train = days_train-days_train[0]

    #N0 = 60e6
    N0 = country_population[country_selector] 
    R0 = confirmed_train[0]
    I0 = R0*10
    S0 = N0-I0-R0
    S0 = N0-I0-R0
    SIR_conditions = S0, I0, R0, N0

#estimate R:
R_prior = initial_R(observation = N_confirmed)
R_prior_ratio = initial_R_ratio(observation = N_confirmed)

#prior for Q:
gamma_list = np.linspace(0.01,1,10) # unlikely that gamma is above 1, would mean infected get removed in less than  a day
beta_list = np.linspace(0.01,1.5,15)
ens_size = 20

Q_prior = ensemble_cov(SIR_conditions,gamma_list,beta_list,ens_size,timesteps=range(0,200))
SIR_conditions_ratio = S0/N0, I0/N0, R0/N0, N0/N0
Q_prior_ratio = ensemble_cov_ratio(SIR_conditions_ratio,gamma_list,beta_list,ens_size,timesteps=range(0,200))

#%%
#SIR Filtering
def SIR_model(y, t, beta, gamma, N0):
    S, I, R = y
    dS_dt = -beta*S*I/N0
    dI_dt = beta*S*I/N0 - gamma*I
    dR_dt = gamma*I
    return dS_dt, dI_dt, dR_dt

def pred_SIR(S0, I0, R0, beta, gamma, days):
    N0 = S0+I0+R0
    x_pred = scipy.integrate.odeint(SIR_model, [S0, I0, R0], days, args=(beta, gamma, N0))
    S_pred, I_pred, R_pred = x_pred[:, 0], x_pred[:, 1], x_pred[:, 2]
    return S_pred, I_pred, R_pred

def pred_next_SIR(S0, I0, R0, beta, gamma):
    tmp = pred_SIR(S0, I0, R0, beta, gamma, [0, 1])
    S1, I1, R1 = tmp[0][1], tmp[1][1], tmp[2][1]
    return S1, I1, R1

def create_F(I, S, N0, beta, gamma):
    F = np.zeros(shape=(3, 3))
    F[0, 0] = (-beta*I/N0)
    F[0, 1] = -beta*S/N0
    F[1, 0] = beta*I/ N0
    F[1, 1] = (beta*S/N0) - gamma
    F[2, 1] = gamma
    F += np.identity(3)
    return F

def calibrate_beta_gamma(x00, x11, beta00, gamma00, ndiv=5000): #ndiv gridsearch #ndiv=100
    S00, I00, R00 = x00
    S11, I11, R11 = x11
    grid_beta, grid_D = np.meshgrid(np.linspace(0.001, 4, ndiv), np.linspace(0.05, 75, ndiv)) #for b, for D, easier than gamma to interpret:4 to 25 been sick
 
    @np.vectorize
    def func(beta, D):
        x10 = pred_next_SIR(S00, I00, R00, beta, 1/D)
        err = np.sum(abs(np.array(x10) - x11))
        # err = abs(np.array(x10[2]) - x11[2])
        return err
    grid_err = func(grid_beta, grid_D)
    idm, idn = np.unravel_index(np.argmin(grid_err, axis=None), grid_err.shape) #argmin tells minimum. idnxidn scalar coordinate
    return grid_beta[idm, idn], 1/grid_D[idm, idn]

def plot_fit(S0, I0, R0, beta, gamma, days, N_confirmed):
    dd = np.linspace(0, days[-1], 1000)
    S_pred, I_pred, R_pred = pred_SIR(S0, I0, R0, beta, gamma, dd)
    fig = plt.figure()
    plt.plot(dd, I_pred, label="I(t)")
    plt.plot(dd, R_pred, label="R(t)")
    plt.grid()
    plt.plot(days, N_confirmed, 'ro', label="Confirmed")
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Number')
    plt.title('SIR model')

    return fig

#optional: Load wuhan data
if 1:
    df = pd.read_csv('data/DXY/wuhan_history.csv')
    days = np.arange(0, len(df))
    #N_confirmed = df['confirm'].to_numpy()
    N_confirmed = df['cum_confirm'].values #alternative
    df['date'] = df['time']

    # whole data
    days_train, confirmed_train = days, N_confirmed
    days_train = days_train-days_train[0]
    #https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/2019-nCoV-outbreak-report-17-01-2020.pdf
    #example case Wuhan:
    #country_selector='Wuhan'
    N0 = 58e6 #hubei
    #N0 = 11.08e6
    R0 = confirmed_train[0]
    I0 = 131
    #I0 = 60
    #I0 = 1700
    S0 = N0-I0-R0
    beta0 = 0.38
    #gamma0 = 0.14
    gamma0 = 0.14

    I00, S00, R00, beta00, gamma00 = I0, S0, R0, beta0, gamma0
    x00 = np.array([S0, I0, R0])

    R_mat = np.diag([280])  
    Q_mat = np.diag([0.1, 0.1, 0.1])
    P00 = np.copy(Q_mat)

#prior covariances
if 1:

    R_mat = R_prior_ratio
    Q_mat = Q_prior_ratio
    P00 = np.diag([1, 1, 1]) 

#other countries
if 1:
    df_global = load_globaldata(country_selector)
    df = df_global
    #cut = 30
    cut = cut #from hybrid approach code
    df_fc = df.iloc[cut:]
    df = df.iloc[cut:-hold_out_test]
    

    days = np.arange(0, len(df))

    N_confirmed = df['cum_confirm'].values #alternative
    N_confirmed_fc = df_fc['cum_confirm'].values #alternative

    if 1: #this aggregates the data
        N_confirmed = df['cum_heal'].values+df['cum_dead'].values+df['cum_confirm'].values
        N_confirmed_fc = df_fc['cum_heal'].values+df_fc['cum_dead'].values+df_fc['cum_confirm'].values


    N_removed = df['cum_heal'].values+df['cum_dead'].values
    df['date'] = df['time']

    days_train, confirmed_train = days, N_confirmed
    days_train = days_train-days_train[0]

    N0 = country_population[country_selector]  
    R0 = confirmed_train[0]
    I0 = R0*10
    S0 = N0-I0-R0
    beta0 = 0.38
    gamma0 = 0.14

    I00, S00, R00, beta00, gamma00 = I0, S0, R0, beta0, gamma0
    x00 = np.array([S0, I0, R0])

# observation matrix
H = np.array([[0, 0, 1]])

I_DA_alt = np.zeros_like(days, dtype=np.float)
I10_DA_alt = np.zeros_like(days, dtype=np.float)
R_DA_alt = np.zeros_like(days, dtype=np.float)
R10_DA_alt = np.zeros_like(days, dtype=np.float)

I_DA = np.zeros_like(days, dtype=np.float)
I10_DA = np.zeros_like(days, dtype=np.float)

R_DA = np.zeros_like(days, dtype=np.float)
S_DA = np.zeros_like(days, dtype=np.float)
yhat_DA = np.zeros_like(days, dtype=np.float)
beta_DA = np.zeros_like(days, dtype=np.float)
gamma_DA = np.zeros_like(days, dtype=np.float)
beta00_DA = np.zeros_like(days, dtype=np.float)
gamma00_DA = np.zeros_like(days, dtype=np.float)

#t+1 forecast
I20_DA = np.append(np.zeros_like(days, dtype=np.float),0)
R20_DA = np.append(np.zeros_like(days, dtype=np.float),0)
S20_DA = np.append(np.zeros_like(days, dtype=np.float),0)

I21_DA = np.append(np.zeros_like(days, dtype=np.float),0)
R21_DA = np.append(np.zeros_like(days, dtype=np.float),0)
S21_DA = np.append(np.zeros_like(days, dtype=np.float),0)

I21_DA_noupdate = np.append(np.zeros_like(days, dtype=np.float),0)
R21_DA_noupdate = np.append(np.zeros_like(days, dtype=np.float),0)
S21_DA_noupdate = np.append(np.zeros_like(days, dtype=np.float),0)

#t+q forecast # out of sample forecasting
q=5 

forecast_horizon = np.zeros(q)

R21_DA = np.append(R21_DA,forecast_horizon)
I_DA = np.append(I_DA,forecast_horizon)
R_DA = np.append(R_DA,forecast_horizon)
S_DA = np.append(S_DA,forecast_horizon)
yhat_DA = np.append(yhat_DA,forecast_horizon)

I10_DA = np.append(I10_DA,forecast_horizon)
S21_DA = np.append(S21_DA,forecast_horizon)
I21_DA = np.append(I21_DA,forecast_horizon)

beta_DA = np.append(beta_DA,forecast_horizon)
gamma_DA = np.append(gamma_DA,forecast_horizon)
beta00_DA = np.append(beta00_DA ,forecast_horizon)
gamma00_DA = np.append(gamma00_DA,forecast_horizon)

# I_DA_alt[day], I10_DA_alt[day], R_DA_alt[day], R10_DA_alt[day]
R10_DA_alt = np.append(R10_DA_alt,forecast_horizon)
I_DA_alt = np.append(I_DA_alt,forecast_horizon)
R_DA_alt = np.append(R_DA_alt,forecast_horizon)
I10_DA_alt = np.append(I10_DA_alt,forecast_horizon)


confirmed_train_m = np.append(confirmed_train,forecast_horizon)
days_train_m = np.array(range(0,len(days_train)+q))

for day, confirmed in tqdm(zip(days_train_m, confirmed_train_m)):
    if day == 0:
        I_DA[day], R_DA[day], beta_DA[day], gamma_DA[day] = I00, R00, beta00, gamma00
    else:
        # new_obs
        if day <= (len(days_train)-1):
            z_t = confirmed

        else:
            z_t = R21_DA[day]

        # predict next x
        S10, I10, R10 = pred_next_SIR(S00, I00, R00, beta00, gamma00)
        x10 = np.array(([S10, I10, R10]))

        # prediction uncertainty
        F1 = create_F(I00, S00, N0, beta00, gamma00)
        P10 = F1 @ P00 @ F1.T + Q_mat


        # forecast observation
        y_tilde = z_t - H @ x10  # fc error
        S1 = H @ P10 @ H.T + R_mat  # error covariace matrix

        # Kalman Gain
        K1 = P10 @ H.T @ np.linalg.inv(S1)

        # updates
        x11 = x10 + K1 @ y_tilde
        S11, I11, R11 = x11
        S11 = N0-R11-I11
        P11 = P10 - K1 @ H @ P10 #p11 bug previously

        # calibrate parameters
        beta11, gamma11 = calibrate_beta_gamma(x00, x11, beta00, gamma00, ndiv=30)

        ######### generate values without updates
        def within_loop(day,S00, I00, R00, P00, beta11, gamma11):
            S10, I10, R10 = pred_next_SIR(S00, I00, R00, beta11, gamma11)
            x10 = np.array(([S10, I10, R10]))

            # prediction uncertainty
            F1 = create_F(I00, S00, N0, beta11, gamma11)
            P10 = F1 @ P00 @ F1.T + Q_mat


            # forecast observation
            y_tilde = z_t - H @ x10  # fc error
            S1 = H @ P10 @ H.T + R_mat  # error covariace matrix

            # Kalman Gain
            K1 = P10 @ H.T @ np.linalg.inv(S1)

            # updates
            x11 = x10 + K1 @ y_tilde
            S11, I11, R11 = x11
            S11 = N0-R11-I11
            P11 = P10 - K1 @ H @ P10 

            #print('within loop',beta11)
            return(I11,I10,R11,R10)

        #fixbeta1,fixgamma1 =  0.1, 0.1
        fixbeta1,fixgamma1 = 0.38, 0.14
        I11_update,I10_update,R11_update,R10_update = within_loop(day,S00, I00, R00, P00, fixbeta1,fixgamma1)

        #take current observation and use new gamma to extrapolate
        y11_tilde = z_t+gamma11*I11

        #save results
        I_DA_alt[day], I10_DA_alt[day], R_DA_alt[day], R10_DA_alt[day] = I11_update , I10_update, R11_update, R10_update
        ###########end generate value without updates

        # save DA result
        S_DA[day],I_DA[day], R_DA[day], I10_DA[day], beta_DA[day], gamma_DA[day], beta00_DA[day], gamma00_DA[day], yhat_DA[day] = S11, I11, R11, I10, beta11, gamma11, beta00, gamma00, y11_tilde

        #generate +1 out of sample forecast
        S21_DA[day+1], I21_DA[day+1], R21_DA[day+1] = pred_next_SIR(S11, I11, R11, beta11, gamma11)

        #reset parameters for next loop
        x00 = np.array([S11, I11, R11])
        P00 = P11 #bug previously
        S00, I00, R00 = x00
        beta00, gamma00 = beta11, gamma11

#construct gamma based forecast
zt1 = N_confirmed_fc+ I_DA*gamma_DA
zt1
zt1 = np.insert(zt1,0,values=0)
zt1 = zt1[:-1]
#Neural SIR model:
S_DAN = S_DA[:-q]
I_DAN = I_DA[:-q]
R_DAN = R_DA[:-q]
#create pandas df
SIR_dict = {'S':S_DAN, 'I':I_DAN, 'R':R_DAN, 'N_conf':N_confirmed_df.values}

df_all = pd.DataFrame(data = SIR_dict, index=N_confirmed_df.index)

#%%
#Univariate LSTM training
#Process Data
N_confirmed_df = df['cum_confirm']
x_train_synth = N_confirmed_df.astype(float)
y_train_synth = N_confirmed_df.astype(float)
print(np.shape(x_train_synth))
x_train = np.expand_dims(x_train_synth,axis=1)#make it one colums, not one row

train = x_train[0:cut] 
test = x_train[cut:]

#normalize the data by scaling it between 0,1.
#De-Meaning makes little sense because it is an exponential process.
#use sklearn scaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


#model setup
model = Sequential()
model.add(LSTM(150,activation="relu",input_shape=(n_input,n_features)))
#model.add(Dropout(0.2))
model.add(Dense(75, activation='relu'))
model.add(Dense(units=1))
#model.add(Activation('softmax'))
#model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

validation_set = np.append(scaled_train[-1],scaled_test)
validation_set = np.expand_dims(validation_set,axis=1)
n_input = 5
n_features = 1
validation_gen = TimeseriesGenerator(data= validation_set,targets=validation_set,length=n_input,batch_size=1)
early_stop = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
model.fit_generator(generator,validation_data=validation_gen,epochs=100,callbacks=[early_stop],steps_per_epoch=10)
pd.DataFrame(model.history.history).plot(title="loss vs epochs curve")

#out of sample prediciton
# holding predictions
test_prediction = []

##last n points from training set
first_eval_batch = scaled_train[-n_input:] #"last batch"
current_batch = first_eval_batch.reshape(1,n_input,n_features)

lstm_fc = 7
for i in range(len(test)+lstm_fc):
    #print(i)
    current_pred = model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

#transform data back to original
true_prediction = scaler.inverse_transform(test_prediction)
true_prediction[:,0]
N_confirmed_df.index = pd.to_datetime(N_confirmed_df.index)
time_series_array = N_confirmed_df.iloc[cut:].index

#predictions
for k in range(0,lstm_fc):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

df_forecast = pd.DataFrame(columns=["confirmed","confirmed_predicted"],index=time_series_array)
df_forecast.loc[:,"confirmed_predicted"] = true_prediction[:,0]
df_forecast.loc[:,"confirmed"] = N_confirmed_df

#show held-out data
df_global_select = df_global[['cum_confirm']]
df_global_select = df_global_select.sum(axis=1).tail(10)
df_global_select.index = pd.to_datetime(df_global_select.index)
df_global_select[-hold_out_test-1:]
df_forecast['confirmed_holdout'] = df_global_select[-hold_out_test-1:]

MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:5]) - np.array(df_forecast["confirmed_predicted"][:5]))/np.array(df_forecast["confirmed"][:5]))
print("MAPE is " + str(MAPE*100) + " %")
MAPE_uni = np.copy(MAPE)

#generate figure
if 0:
    fig= plt.figure(figsize=(10,5))
    plt.title("{} - Results".format(country))
    plt.plot(df_forecast.index,df_forecast["confirmed"]/scale,label="confirmed")
    plt.plot(df_forecast.index,df_forecast["confirmed_predicted"]/scale,label="confirmed_predicted")
    plt.legend()
    plt.show()

#%%
#In Sample Forecast performance
if 1 :
    test_prediction = []
    lstm_fc = 7 #fc periods
    for i in range(len(N_confirmed)+lstm_fc):

        if i < len(scaled_train)-n_input:
        first_eval_batch = scaled_train[i:i+n_input] #"last batch"
        current_batch = first_eval_batch.reshape(1,n_input,n_features)

        current_pred = model.predict(current_batch)[0]
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        #at the end of the sample advance the last batch and move forwad
        else:
        #current_batch
        current_pred = model.predict(current_batch)[0]
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    #now put predictions into a dataframe
    true_prediction = scaler.inverse_transform(test_prediction)
    df_sliding = pd.DataFrame(true_prediction[:-lstm_fc])
    df_sliding.index = N_confirmed_df.index

    df_sliding['N_conf'] = N_confirmed_df
    lstm_uni_n_conf = df_sliding.iloc[:,0:1]

    #error estimate
    error_sliding1_abs = np.abs(df_sliding[0]-df_sliding['N_conf'])
    error_sliding1_mape = sum(np.abs((df_sliding[0]-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
    error_sliding1 = sum(np.abs(df_sliding[0]-df_sliding['N_conf']))/len(df_sliding) #mae
    error_sliding1_sq = np.square((df_sliding[0]-df_sliding['N_conf']))
    error_sliding1_sqrt = np.sqrt(error_sliding1_sq.astype(float))

    error_sliding1_sq = sum(error_sliding1_sq)/len(error_sliding1_sq) #mse
    error_sliding1_sqrt = sum(error_sliding1_sqrt)/len(error_sliding1_sqrt) #rmse

#%%
#Couple LSTM and SIR model
df_all = pd.DataFrame(data = SIR_dict, index=N_confirmed_df.index)
x_train_synth = df_all
y_train_synth = N_confirmed.astype(float)

cut = len(x_train)-5  #adjust according to out of sample training
train = x_train[0:cut] 
test = x_train[cut:]

#scale data
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

n_input = 5
n_features = np.shape(df_all)[1]
generator = TimeseriesGenerator(data= scaled_train,targets=scaled_train,length=n_input,batch_size=1)

#model specification
model = Sequential()
model.add(LSTM(150,activation="relu",input_shape=(n_input,n_features)))
#model.add(Dropout(0.2))
model.add(Dense(75, activation='relu'))
model.add(Dense(units=4))
#model.add(Activation('softmax'))
#model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

#check model summary
model.summary()

#create validation set
validation_set = np.append(scaled_train[-1],scaled_test)
validation_set = validation_set.reshape(n_input+1,n_features)
n_input = 5
n_features_val = 1
validation_gen = TimeseriesGenerator(data= validation_set,targets=validation_set,length=n_input,batch_size=1)

#train model
early_stop = EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
x1,y1 = validation_gen[11]
model.fit_generator(generator,validation_data=validation_gen,epochs=100,callbacks=[early_stop],steps_per_epoch=10)
#plot training and test accuracy
#pd.DataFrame(model.history.history).plot(title="loss vs epochs curve")

#create predictions
test_prediction = []
##last n points from training set
first_eval_batch = scaled_train[-n_input:] #"last batch"
current_batch = first_eval_batch.reshape(1,n_input,n_features)

for i in range(len(test)+lstm_fc):
    current_pred = model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

#transform data back to original
true_prediction = scaler.inverse_transform(test_prediction)
var=3
true_prediction[:,var] 

#adjust datetime index
N_confirmed_df.index = pd.to_datetime(N_confirmed_df.index)
time_series_array = N_confirmed_df.iloc[cut:].index
for k in range(0,lstm_fc):
    time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

df_forecast = pd.DataFrame(columns=["confirmed","confirmed_predicted"],index=time_series_array)

#show held-out data
df_global_select = df_global[['cum_confirm']]
df_global_select = df_global_select.sum(axis=1).tail(10)

df_global_select.index = pd.to_datetime(df_global_select.index)
df_global_select[-hold_out_test-1:]
df_forecast['confirmed_holdout'] = df_global_select[-hold_out_test-1:]
MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:5]) - np.array(df_forecast["confirmed_predicted"][:5]))/np.array(df_forecast["confirmed"][:5]))
print("MAPE is " + str(MAPE*100) + " %")
MAPE_SIR = np.copy(MAPE)

#plot data
if 0:
    fig= plt.figure(figsize=(10,5))
    plt.title("{} - Results".format(country))
    plt.plot(df_forecast.index,df_forecast["confirmed"]/scale,label="confirmed")
    plt.plot(df_forecast.index,df_forecast["confirmed_predicted"]/scale,label="confirmed_predicted")
    plt.legend()
    plt.show()

#in sample fit
if 1 :
    test_prediction = []
    lstm_fc = 7 #prediction range
    for i in range(len(N_confirmed)+lstm_fc):

        if i < len(scaled_train)-n_input:
        
        first_eval_batch = scaled_train[i:i+n_input] #"last batch"
        print(i)
        print(first_eval_batch)
        current_batch = first_eval_batch.reshape(1,n_input,n_features)
        
        current_pred = model.predict(current_batch)[0]
        print(current_pred)
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        #at the end of the sample advance the last batch and move forward
        
        else:
        #current_batch
        current_pred = model.predict(current_batch)[0]
        print(i)
        print(current_batch)
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    if 1:
        true_prediction = scaler.inverse_transform(test_prediction)
        df_sliding = pd.DataFrame(true_prediction[:-lstm_fc-1])
        df_sliding.index = N_confirmed_df[1:].index

    df_sliding['N_conf'] = N_confirmed_df

#construct fc metrics
df_sliding = df_sliding.rename(columns={0: "S", 1: "I", 2:"R", 3:"N_conf_input"})
error_sliding2 = sum(df_sliding['N_conf_input']-df_sliding['N_conf'])/len(df_sliding)
error_sliding2_abs = np.abs(df_sliding['N_conf_input']-df_sliding['N_conf'])
error_sliding2_mape = sum(np.abs((df_sliding['N_conf_input']-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
error_sliding2 = sum(np.abs(df_sliding['N_conf_input']-df_sliding['N_conf']))/len(df_sliding)
error_sliding2_sq = np.square((df_sliding['N_conf_input']-df_sliding['N_conf']))
error_sliding2_sqrt = np.sqrt(error_sliding2_sq.astype(float))
#mean
error_sliding2_sq = sum(error_sliding2_sq)/len(error_sliding2_sq)
error_sliding2_sqrt = sum(error_sliding2_sqrt)/len(error_sliding2_sqrt)
lstm_sir_n_conf = df_sliding['N_conf_input']

#compute additional metrics:
error_sliding3 = sum(np.abs(df_sliding['zt1']-df_sliding['N_conf']))/len(df_sliding)
error_sliding3_mape = sum(np.abs((df_sliding['zt1']-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
error_sliding3_sq = np.square((df_sliding['zt1']-df_sliding['N_conf']))
error_sliding3_sqrt = np.sqrt(error_sliding3_sq.astype(float))
error_sliding3_sq = sum(error_sliding3_sq)/len(error_sliding3_sq)
error_sliding3_sqrt = sum(error_sliding3_sqrt)/len(error_sliding3_sqrt)
#error estimate
error_sliding4_abs = np.abs(df_sliding['R_DA']-df_sliding['N_conf'])
error_sliding4 = sum(np.abs(df_sliding['R_DA']-df_sliding['N_conf']))/len(df_sliding)
error_sliding4_mape = sum(np.abs((df_sliding['R_DA']-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
error_sliding4_sq = np.square((df_sliding['R_DA']-df_sliding['N_conf']))
error_sliding4_sqrt = np.sqrt(error_sliding4_sq.astype(float))
error_sliding4_sq = sum(error_sliding4_sq)/len(error_sliding4_sq)
error_sliding4_sqrt = sum(error_sliding4_sqrt)/len(error_sliding4_sqrt)
#Model Average1
error_sliding5 = sum(np.abs(df_sliding['M_avg1']-df_sliding['N_conf']))/len(df_sliding)
error_sliding5_mape = sum(np.abs((df_sliding['M_avg1']-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
error_sliding5_sq = np.square((df_sliding['M_avg1']-df_sliding['N_conf']))
error_sliding5_sqrt = np.sqrt(error_sliding5_sq.astype(float))
error_sliding5_sq = sum(error_sliding5_sq)/len(error_sliding5_sq)
error_sliding5_sqrt = sum(error_sliding5_sqrt)/len(error_sliding5_sqrt)
#Model Average2
error_sliding6 = sum(np.abs(df_sliding['M_avg2']-df_sliding['N_conf']))/len(df_sliding)
error_sliding6_mape = sum(np.abs((df_sliding['M_avg2']-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
error_sliding6_sq = np.square((df_sliding['M_avg2']-df_sliding['N_conf']))
error_sliding6_sqrt = np.sqrt(error_sliding6_sq.astype(float))
error_sliding6_sq = sum(error_sliding6_sq)/len(error_sliding6_sq)
error_sliding6_sqrt = sum(error_sliding6_sqrt)/len(error_sliding6_sqrt)

#Dynamic Model averaging:
sumerror = error_sliding2_abs+error_sliding4_abs # total error
ew1 = error_sliding2_abs/sumerror #LSTM constribution to error 
ew2 = error_sliding4_abs/sumerror  #R_DA contribution to error

#print(w1,w2)
m_avg = df_sliding['R_DA']*ew1 + df_sliding['N_conf_input']*ew2

#Model Average2
error_sliding7 = sum(np.abs(m_avg-df_sliding['N_conf']))/len(df_sliding)
error_sliding7_mape = sum(np.abs((m_avg-df_sliding['N_conf'])/df_sliding['N_conf']))    /    len(df_sliding)
error_sliding7_sq = np.square((m_avg-df_sliding['N_conf']))
error_sliding7_sqrt = np.sqrt(error_sliding7_sq.astype(float))
error_sliding7_sq = sum(error_sliding7_sq)/len(error_sliding7_sq)
error_sliding7_sqrt = sum(error_sliding7_sqrt)/len(error_sliding7_sqrt)

# Compare all Approaches
if 0:
    plottil = 50
    plt.title(country_selector)
    plt.plot(lstm_uni_n_conf[:plottil]/scale,'b-',label='LSTM',linestyle='-')
    plt.plot(m_avg[:plottil]/scale,label='Neural SIR',linestyle='solid')
    plt.plot(df_sliding['N_conf'][0:plottil]/scale,'ro',label='Conf. Cases')
    plt.plot(df_sliding['R_DA'][0:plottil]/scale,'k.-',label='SIR Model')
    plt.ylabel('Percentage of Population')
    plt.legend()
    plt.tight_layout(pad=1.21)
    ax = plt.gca()
    ax.xaxis.set_tick_params(rotation=60, labelsize=8)
    #plt.savefig(str(country_selector+'LSTM_comparison.pdf'))
    plt.show()
