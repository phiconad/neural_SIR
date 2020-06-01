import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from SITR import load_globaldata

import os
os.chdir()


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



def calibrate_beta_gamma(x00, x11, beta00, gamma00, ndiv=100): #ndiv gridsearch
    S00, I00, R00 = x00
    S11, I11, R11 = x11
    grid_beta, grid_D = np.meshgrid(np.linspace(0.1, 1, ndiv), np.linspace(4, 25, ndiv)) 
    @np.vectorize
    def func(beta, D):
        x10 = pred_next_SIR(S00, I00, R00, beta, 1/D)
        err = np.sum(abs(np.array(x10) - x11))
        return err
    grid_err = func(grid_beta, grid_D)
    idm, idn = np.unravel_index(np.argmin(grid_err, axis=None), grid_err.shape) 
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


#Load wuhan data
df = pd.read_csv('data/DXY/wuhan_history.csv')
days = np.arange(0, len(df))
N_confirmed = df['cum_confirm'].values #alternative
df['date'] = df['time']






# whole data
days_train, confirmed_train = days, N_confirmed
days_train = days_train-days_train[0]



#country_selector='Wuhan'
N0 = 58e6 #hubei
#N0 = 11.08e6
R0 = confirmed_train[0]
I0 = 131
#I0 = 60
#I0 = 1700
S0 = N0-I0-R0
beta0 = 0.38
gamma0 = 0.14

I00, S00, R00, beta00, gamma00 = I0, S0, R0, beta0, gamma0
x00 = np.array([S0, I0, R0])



#prior covariances
if 1:

    R_mat = R_prior_ratio
    Q_mat = Q_prior_ratio

    P00 = np.copy(Q_mat)


#other countries
if 1:
    country_selector='France'
    df_global = load_globaldata(country_selector)
    df = df_global
    cut = 10
    df = df.iloc[cut:]

    days = np.arange(0, len(df))

    N_confirmed = df['cum_confirm'].values #alternative
    N_removed = df['cum_heal']+df['cum_dead']
    df['date'] = df['time']

    days_train, confirmed_train = days, N_confirmed
    days_train = days_train-days_train[0]

    N0 = 66e6 
    R0 = confirmed_train[0]
    I0 = R0*10
    S0 = N0-I0-R0
    beta0 = 0.38
    gamma0 = 0.07

    I00, S00, R00, beta00, gamma00 = I0, S0, R0, beta0, gamma0
    x00 = np.array([S0, I0, R0])



# observation matrix
H = np.array([[0, 0, 1]])


#create containers
I_DA_alt = np.zeros_like(days, dtype=np.float)
I10_DA_alt = np.zeros_like(days, dtype=np.float)
R_DA_alt = np.zeros_like(days, dtype=np.float)
R10_DA_alt = np.zeros_like(days, dtype=np.float)

I_DA = np.zeros_like(days, dtype=np.float)
I10_DA = np.zeros_like(days, dtype=np.float)

R_DA = np.zeros_like(days, dtype=np.float)
S_DA = np.zeros_like(days, dtype=np.float)
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

#t+q forecast
q=120 #300
forecast_horizon = np.zeros(q)

R21_DA = np.append(R21_DA,forecast_horizon)
I_DA = np.append(I_DA,forecast_horizon)
R_DA = np.append(R_DA,forecast_horizon)
I10_DA = np.append(I10_DA,forecast_horizon)
S21_DA = np.append(S21_DA,forecast_horizon)
I21_DA = np.append(I21_DA,forecast_horizon)

beta_DA = np.append(beta_DA,forecast_horizon)
gamma_DA = np.append(gamma_DA,forecast_horizon)
beta00_DA = np.append(beta00_DA ,forecast_horizon)
gamma00_DA = np.append(gamma00_DA,forecast_horizon)

R10_DA_alt = np.append(R10_DA_alt,forecast_horizon)
I_DA_alt = np.append(I_DA_alt,forecast_horizon)
R_DA_alt = np.append(R_DA_alt,forecast_horizon)
I10_DA_alt = np.append(I10_DA_alt,forecast_horizon)



#Start Assimilation


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
        P11 = P10 - K1 @ H @ P10 

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


            return(I11,I10,R11,R10)

        #fixbeta1,fixgamma1 =  0.1, 0.1
        fixbeta1,fixgamma1 = 0.38, 0.14
        I11_update,I10_update,R11_update,R10_update = within_loop(day,S00, I00, R00, P00, fixbeta1,fixgamma1)



        #save results
        I_DA_alt[day], I10_DA_alt[day], R_DA_alt[day], R10_DA_alt[day] = I11_update , I10_update, R11_update, R10_update
        ###########end generate value without updates

        # save DA result
        I_DA[day], R_DA[day], I10_DA[day], beta_DA[day], gamma_DA[day], beta00_DA[day], gamma00_DA[day] = I11, R11, I10, beta11, gamma11, beta00, gamma00

        #generate +1 out of sample forecast
        S21_DA[day+1], I21_DA[day+1], R21_DA[day+1] = pred_next_SIR(S11, I11, R11, beta11, gamma11)

        #generate +1 out of sample forecast old params
        #S20_DA[day+1], I20_DA[day+1], R20_DA[day+1] = pred_next_SIR(S11, I11, R11, beta00, gamma00)


        #reset parameters for next loop
        x00 = np.array([S11, I11, R11])
        P00 = P11 #bug previously
        S00, I00, R00 = x00
        beta00, gamma00 = beta11, gamma11




# simple_SIR plots

if 1:

    longdate = pd.read_excel('datevector_excel.xls',squeeze=True)
    longdate = longdate[7:]

    # simple_SIR very long forecast updated parameters
    plotuntil = 147
    normalizer = N0
    scale = N0/100 #for percentage points
    plt.figure()
    plt.plot(longdate[cut:cut+len(N_confirmed)], N_confirmed[0:]/scale, 'ro', label="Observed Confirmed")
    plt.plot(longdate[cut:cut+plotuntil], R_DA[0:plotuntil]/scale, 'k.-', label='Recovered')
    plt.plot(longdate[cut:cut+plotuntil], I_DA[0:plotuntil]/scale, label='Infected')
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_tick_params(rotation=60, labelsize=8)
    #plt.xticks(np.arange(0, 25+220,20))
    plt.ylabel('Percentage of Population')
    plt.title(country_selector)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SIR_global_long.pdf')
    plt.show()

    # simple_SIR short run
    plotuntil = 33
    normalizer = N0
    scale = N0/100 #for percentage points
    plt.figure()
    plt.plot(longdate[0:len(N_confirmed)], N_confirmed[0:]/scale, 'ro', label="Observed Confirmed")
    #plt.plot(longdate[0:len(N_removed)], N_removed[0:]/scale, '+', label="Removed")#
    plt.plot(longdate[0:plotuntil], R_DA[0:plotuntil]/scale, 'k.-', label='Recovered')
    plt.plot(longdate[0:plotuntil], I_DA[0:plotuntil]/scale, label='Infected')
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_tick_params(rotation=60, labelsize=8)
    #plt.xticks(np.arange(0, 25+220,20))
    plt.ylabel('Percentage of Population')
    plt.title(country_selector)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SIR_global_short.pdf')
    plt.show()

    # simple_SIR medium
    plotuntil = 50
    normalizer = N0
    scale = N0/100 #for percentage points
    plt.figure()
    plt.plot(longdate[0:len(N_confirmed)], N_confirmed[0:]/scale, 'ro', label="Observed Confirmed")
    #plt.plot(longdate[0:len(N_removed)], N_removed[0:]/scale, '+', label="Removed")#
    plt.plot(longdate[0:plotuntil], R_DA[0:plotuntil]/scale, 'k.-', label='Recovered')
    plt.plot(longdate[0:plotuntil], I_DA[0:plotuntil]/scale, label='Infected')
    ax = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_tick_params(rotation=60, labelsize=8)
    #plt.xticks(np.arange(0, 25+220,20))
    plt.ylabel('Percentage of Population')
    plt.title(country_selector)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SIR_global_medium.pdf')
    plt.show()


if 0:

    plt.figure()
    plt.plot(df['date'], N_confirmed[:len(df['date'])], 'ro', label="Confirmed")
    plt.plot(df['date'], R_DA[:len(df['date'])], 'k.-', label='R_DA')
    plt.plot(df['date'], I_DA[:len(df['date'])], label='I_DA')
    ax = plt.gca()
    ax.xaxis.set_tick_params(rotation=60, labelsize=8)
    plt.xticks(np.arange(0, 15)*2)
    plt.legend()
    plt.show()

    #plot  simple_SIR scaled
    normalizer = N0
    scale = N0/100 #for percentage points
    plt.figure()
    plt.plot(df['date'], N_confirmed/scale, 'ro', label="Confirmed")
    plt.plot(df['date'], R_DA/scale, 'k.-', label='R_DA')
    plt.plot(df['date'], I_DA/scale, label='I_DA')
    ax = plt.gca()
    ax.xaxis.set_tick_params(rotation=60, labelsize=8)
    plt.xticks(np.arange(0, 15)*2)
    plt.ylabel('Percentage of Population')
    plt.legend()
    plt.show()
