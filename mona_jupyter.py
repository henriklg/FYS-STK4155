from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error



X = np.arange(0, 1, 0.004)
Y = np.arange(0, 1, 0.004)
n = np.size(X, 0)


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4




def FindData(x , y):
    #Franke funksjonen trenger et to dimensjonalt nett av variabler, meshgrid:
    x_d, y_d = np.meshgrid(x,y)
    z_d = FrankeFunction(x_d, y_d) #+ 0.05*np.random.randn(n,n)  #+random noise  0.05*np.random.randn(100, 100)
    return x_d, y_d, z_d


def convertData(x_d, y_d, z_d):
    #Strekker ut matrisene
    x = x_d.reshape(-1,1)
    y = y_d.reshape(-1,1)
    z = z_d.reshape(-1,1)

    # returnerer også lengden (antall rader) til den nye utstrakte vektoren
    n = np.size(x, 0)
    return (x, y, z, n)



def plotSurface(x_d, y_d, z_d):
    # Kode for å plotte overflater, basert på oppgaveteksten
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x_d, y_d, z_d, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z - Franke')

    # Add a color bar which maps values to colors.
    clb = fig.colorbar(surf, shrink=0.5, aspect=5)
    clb.ax.set_title('Level')

    plt.show()


def XY(x,y):
    # Oppretter designmatrisen/Vandermondes
    xyb = np.c_[np.ones((x.shape)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3,y**4,
           x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]
    #print(xyb.shape)
    return xyb


#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
def XYsklearn(degreePoly, X,Y):

    # Oppretter meshgrid av datasettet
    x_d, y_d, z_d = FindData(X , Y)
    # Plotter z_d med meshgrid av x og y
    plotSurface(x_d, y_d, z_d)

    # Konverterer meshgrid til kolonnevektorer
    x, y, z, m = convertData(x_d, y_d, z_d)

    # Oppretter independent variabel matrise av x og y vektorene
    independent_multi = [x,y]
    #np.c_[x, y]

    # Bruker sklearn for å finne Lasso-modell. Bruker PolynomialFeatures for ikke lineær funksjon
    poly2 = PolynomialFeatures(degree=degreePoly)
    # Transformerer variabelmatrisen
    poly_sklearn = poly2.fit_transform(independent_multi)

    # Beregner Lasso modell
    #lasso=linear_model.Lasso(alpha=0.001)
    #lasso.fit(Lasso_sklearn, z_Lasso)

    # beregner forventet z-verdier av x og y verdier
    #predictLasso = lasso.predict(Lasso_sklearn)

    return poly_sklearn




def beta_model(model, xyb, z):
    if (model == 'Linear'):
        # betaLinear:
        betaLinear = np.linalg.inv(xyb.T.dot(xyb)).dot(xyb.T).dot(z)
        return betaLinear
    elif (model == 'Ridge'):
        I = np.identity(np.size(xyb, 1))
        lambda_R = 0.5
        betaRidge = np.linalg.inv(xyb.T.dot(xyb) + lambda_R*(I)).dot(xyb.T).dot(z)
        return betaRidge
    elif (model == 'Lasso'):
        #betaLasso =
        return betaLasso

#trener modellen:
def predict(xyb,beta,n):
    # Finner utregnet forventet model.
    zpredict = xyb.dot(beta)

    # shapen til x og y må ha like mange punkter, bruker derfor x_d.shape = y_d.shape
    zpredict_mesh = zpredict.reshape(n,n)
    return zpredict, zpredict_mesh


# #### Statistical


def mu(z,n):
    #gjennomsnittsverdien
    z_mean = (1/n ) * np.sum(z)
    return z_mean

def calc_Variance(z, z_mu,n):
    #Sample variance:
    var_z = (1/n)* sum((z-z_mu)**2)
    return var_z

def MSE(z, z_tilde, n):
    #Mean Squared Error: z = true value, z_tilde = forventet z utifra modell
    MSE = (1/n)*(sum(z-z_tilde)**2)
    #error = np.mean( np.mean((z - z_tilde)**2, axis=1, keepdims=True) )

    return MSE

def calc_R_2(z, z_tilde, z_mean, n):
    R_2 = 1- ((sum(z.reshape(-1,1)-z_tilde)**2)/(sum((z-z_mean)**2)))
    return R_2

def MSEk(z, z_tilde, n):
    MSE = (1/n)*(sum((z-z_tilde)**2))
    return MSE




def sigma_2(xyb, z_true, z_predict, N, p):
    sigma2 = (1/(N-(p-1 )))* (sum(z_true-z_predict)**2)

    varBeta = np.linalg.inv(xyb.T.dot(xyb))* sigma2#**2

    #Intervall betacoefisienter:
    s_beta_c_int = np.sqrt(np.diag(varBeta))

    return  s_beta_c_int




# Finner Confidens Intervallet
def confidenceIntervall(z_true, z_predict, N, p, xyb, beta):
    #N = z punkter , p = ant polynom.

    sigma2 = (1/(N-(p-1 ))) * (sum((z_true-z_predict)**2))
    # Betas varians:
    varBeta = np.linalg.inv((xyb.T.dot(xyb)))* sigma2

    # estimert standardavvik pr beta.
    # Må forklares i rapport hvorfor man velger å gjøre det slik og ikke på andre måter.
    betaCoeff = (np.sqrt(np.diag(varBeta))).reshape(-1,1)
    #Intervall betacoefisienter:
    beta_confInt = np.c_[beta-betaCoeff, beta+betaCoeff]
    return beta, betaCoeff, beta_confInt



def confidenceIntervall2(z_true, N, p, xyb, beta):
    #N = z punkter , p = ant polynom.
    z_mean = (1/N ) * np.sum(z_true)
    sigma2 = (1/(N-(p-1 ))) * (sum((z_true-z_mean)**2))
    # Betas varians:
    varBeta = np.linalg.inv((xyb.T.dot(xyb)))* sigma2

    # estimert standardavvik pr beta.
    # Må forklares i rapport hvorfor man velger å gjøre det slik og ikke på andre måter.
    betaCoeff = (np.sqrt(np.diag(varBeta))).reshape(-1,1)
    #Intervall betacoefisienter:
    beta_confInt = np.c_[beta-betaCoeff, beta+betaCoeff]
    return beta, betaCoeff, beta_confInt



# Finner modellens bias og varians:
def bias(z_true, z_predict, N):
    bias2 = np.sum((z_true - np.mean(z_predict))**2 )/N
    #bias = np.mean( (z_true - np.mean(z_predict, axis=1, keepdims=True))**2 )
    return bias2

def var2(z_predict, N):
    var = np.sum((z_predict - np.mean(z_predict))**2)/N
    #variance = np.mean( np.var(z_predict, axis=1, keepdims=True) )
    return var


# ### Oppretter train/testdata:


# deler datasettet i tre. en testdel og resten train. 30/70
def test2(x,y,z, model):
    # Oppretter en matrise av x, y og z. Datasettet deles deretter i test og train.
    data1 = np.concatenate((x, y), axis = 1)
    #data2 = z
    # Splitter datasettet i to test og train .Test er et valideringssett og

    x_train, x_test, y_train, y_test = train_test_split(data1, z, test_size=0.2)

    #x_test og y_test er valideringspunktene som skal brukes for å finne biasene.
    # Henter validerings designmatrisa:

    xybvalidering = XY(x_test[:,0],x_test[:,1])
    print(xybvalidering.shape)

    data = np.concatenate((x_train, y_train), axis = 1)
    #print(data.shape)
    n = np.size(data, 0) #rader
    m = np.size(data, 1) #kolonner

    # Shuffle dataset:
    np.random.shuffle(data)

    # datasettet deles i 3 - en del test og to treningsdel
    split_dataset = np.array_split(data,3)
    test = split_dataset[0]
    train = np.concatenate((split_dataset[1], split_dataset[2]), axis = 0)

    # TRENING:
    # Når hele datasettet er gått igjennom en hel iterasjon så trener man på train settet
    #1. Finner designmatrisen, xyb:
    xyb_train = XY(train[:,0],train[:,1])

    #2. Finner beta av treningsettet med treningsettets designmatrise og treningsdelens z-verdier.
    beta = beta_model(model, xyb_train, train[:,2])

    # TEST:
    # Designmatrisen opprettes for testsettet
    xyb_test = XY(test[:,0],test[:,1])

    nr =  np.size(test[:,2],0)


    # Finner forventet z_verdier for testsettet ved å bruke utregnet modell, beta, over og testsettets x og y verdier fra xyb-matrisa.
    zpredict_test = xyb_test.dot(beta)
    zpredict_test_validering = xybvalidering.dot(beta)
    print(zpredict_test_validering.shape)
    Nval = np.size(zpredict_test_validering, 0)
    # regner ut mean squared error
    mse = MSEk(test[:,2], zpredict_test, nr)

    z_mean = (1/nr ) * np.sum(test[:,2])

    # Regner ut R_2 score:
    R_2 = 1 - (((sum(test[:,2]-zpredict_test))**2)/(sum((test[:,2]-z_mean)**2)))

    betaCoeff = sigma_2(xyb_test,test[:,2], zpredict_test, nr, 5)
    #Intervall betacoefisienter:
    beta_confInt = np.c_[beta-betaCoeff, beta+betaCoeff]

    #bias: y_test er valideringspunktene z.
    biasModel = np.mean((y_test - (zpredict_test_validering))**2 )

    #bias(y_test, zpredict_test_validering, Nval)
    #varians:
    var = var2(zpredict_test, nr)
    #plt.scatter(test[:,0], test[:,1], test[:,2], label='Data points')
    #error = np.mean(np.mean((test[:,2] - zpredict_test)**2, axis=1, keepdims=True) )

    #plt.plot(test[:,0], test[:,2], label='f(x)')
    #plt.scatter(x_test, y_test, label='Data points')
    #plt.scatter(x_test, np.mean(y_pred, axis=1), label='Pred')

    #plt.scatter(test[:,0::10000], test[:,2::10000], label='Data points')
    #plt.scatter(test[:,0::10000], np.mean(zpredict_test), label='Pred')
    #plt.show()
    #plt.scatter(test[:,0], test[:,1], np.mean(zpredict_test, axis=1), label='Pred')

    #print (beta_confInt )
    return mse, R_2, biasModel, var, beta_confInt

# deler datasettet i tre. en testdel og resten train. 30/70
def test(x,y,z, model):
    # Oppretter en matrise av x, y og z. Datasettet deles deretter i test og train.
    data = np.concatenate((x, y, z), axis = 1)

    n = np.size(data, 0) #rader
    m = np.size(data, 1) #kolonner

    # Shuffle dataset:
    np.random.shuffle(data)

    # datasettet deles i 3 - en del test og to treningsdel
    split_dataset = np.array_split(data,3)
    test = split_dataset[0]
    train = np.concatenate((split_dataset[1], split_dataset[2]), axis = 0)

    # TRENING:
    # Når hele datasettet er gått igjennom en hel iterasjon så trener man på train settet
    #1. Finner designmatrisen, xyb:
    xyb_train = XY(train[:,0],train[:,1])

    #2. Finner beta av treningsettet med treningsettets designmatrise og treningsdelens z-verdier.
    beta = beta_model(model, xyb_train, train[:,2])

    # TEST:
    # Designmatrisen opprettes for testsettet
    xyb_test = XY(test[:,0],test[:,1])

    nr =  np.size(test[:,2],0)

    # Finner forventet z_verdier for testsettet ved å bruke utregnet modell, beta, over og testsettets x og y verdier fra xyb-matrisa.
    zpredict_test = xyb_test.dot(beta)

    # regner ut mean squared error
    mse = MSEk(test[:,2], zpredict_test, nr)

    z_mean = (1/nr ) * np.sum(test[:,2])

    # Regner ut R_2 score:
    R_2 = 1 - (((sum(test[:,2]-zpredict_test))**2)/(sum((test[:,2]-z_mean)**2)))

    betaCoeff = sigma_2(xyb_test,test[:,2], zpredict_test, nr, 5)
    #Intervall betacoefisienter:
    beta_confInt = np.c_[beta-betaCoeff, beta+betaCoeff]

    #bias:
    biasModel = bias(test[:,2], zpredict_test, nr)
    #varians:

    #calc_Variance

    var = var2(zpredict_test, nr)
    #plt.scatter(test[:,0], test[:,1], test[:,2], label='Data points')
    #error = np.mean(np.mean((test[:,2] - zpredict_test)**2, axis=1, keepdims=True) )

    #plt.plot(test[:,0], test[:,2], label='f(x)')
    #plt.scatter(x_test, y_test, label='Data points')
    #plt.scatter(x_test, np.mean(y_pred, axis=1), label='Pred')

    #plt.scatter(test[:,0::10000], test[:,2::10000], label='Data points')
    #plt.scatter(test[:,0::10000], np.mean(zpredict_test), label='Pred')
    #plt.show()
    #plt.scatter(test[:,0], test[:,1], np.mean(zpredict_test, axis=1), label='Pred')

    #print (beta_confInt )
    return mse, R_2, biasModel, var, beta_confInt


# ### Cross-Validation
# #### K-fold


def test_k_fold(x,y,z, k, model):
    # Oppretter en matrise av x, y og z. Datasettet som skal crossvalideres
    size =(k,1)
    biasModel = np.zeros(size)
    data1 = np.concatenate((x, y), axis = 1)
    #data2 = z
    # Splitter datasettet i to test og train .Test i dette tilfellet er et valideringssett og inndelingen:
    #25% validering og resten train som videre deles inn i 25% test og resten treining for å finne beta.

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data1, z, test_size=0.25)

    #x_test og y_test er valideringspunktene som skal brukes for å finne biasene.
    # Henter validerings designmatrisa:
    xybvalidering = XY(x_test[:,0],x_test[:,1])
    #print(xybvalidering.shape)

    data = np.concatenate((x_train, y_train), axis = 1)

    n = np.size(data, 0) #rader
    m = np.size(data, 1) #kolonner

    # Shuffle dataset:
    np.random.shuffle(data)

    mse = 0
    r2 = 0

    betaCoeff = 0
    bias = 0
    varianceModel = 0
    biasModelk = 0

    #splitter dataset i k like deler. Oppretter antall folder. Bruker array_split. Da blir k deler blir ikke eksakt like,
    #men tilnærmet like, stort datasett gir lite forskjell.
    split_dataset = np.array_split(data,k)
    #Definerer alle k gangene datasettet skal deles opp.
    for i in range(0, k):
        # resetter train før hver oppdeling av datasettet.
        train = np.zeros((0,3))

        # j definerer selve splittdelene og man bestemmer her om del j er en trenings eller test del.
        for j in range(0,k):
            if j == i:
                test = split_dataset[j]
                continue
            train = np.concatenate((train, split_dataset[j]), axis = 0)
        #print(train.shape)

        # TRENING:
        # Når hele datasettet er gått igjennom en hel iterasjon så trener man på train settet
        #1. Finner designmatrisen, xyb:
        xyb_train = XY(train[:,0],train[:,1])

        #2. Finner beta av treningsettet med treningsettets designmatrise og treningsdelens z-verdier.
        # finner beta for sett nr i
        beta = beta_model(model, xyb_train, train[:,2])

        # VALIDERING
        # Finner Z_k- denne skal sjekkes mot z_validering og regne ut den totalebiasen etterhvert
        zpredict_test_validering = xybvalidering.dot(beta)
        #print(zpredict_test_validering.shape)
        Nval = np.size(zpredict_test_validering, 0)

        # TEST:
        # Designmatrisen opprettes for testsettet
        xyb_test = XY(test[:,0],test[:,1])

        nr =  np.size(test[:,2],0)

        # Finner forventet z_verdier for testsettet ved å bruke utregnet modell, beta, over og testsettets x og y verdier fra xyb-matrisa.
        zpredict_test = xyb_test.dot(beta)
        #zTruek[i] = test[:,2]
        #print(test[:,2].shape)
        #np.concatenate((zpredict_test[i], ), axis=0)

        # regner ut mean squared error pr fold
        #part_mse = MSEk(test[:,2], zpredict_test, nr)
        #FOR VALIDERINGSSETTET:
        # Finner part error pr fold. Førs finne meansquared pr z og deretter mean av dette:
        part_mse = np.mean(MSEk(y_test, zpredict_test_validering, Nval))

        #For testsettet:
        #part_mse = np.mean(MSEk(test[:,2], zpredict_test, nr))


        print(part_mse)



#       part_mse =
        #part_mse = MSE(test[:,2], zpredict_test, nr)
        #error
        #part_mse = np.mean(np.mean((test[:,2] - zpredict_test)**2) )

        mse = mse + part_mse

        z_mean = (1/nr ) * np.sum(test[:,2])

        # Regner ut R_2 score pr fold:
        R_2 = 1 - (((sum(test[:,2]-zpredict_test))**2)/(sum((test[:,2]-z_mean)**2)))
        r2 = r2 + R_2

        #Finner betacoeffisientenes varians
        betaCoeffpart = sigma_2(xyb_test,test[:,2], zpredict_test, nr, 5)

        betaCoeff = betaCoeff + betaCoeffpart

        #bias:

        #varians:
        #var = var2(zpredict_test, nr)
        varpart = var2(zpredict_test_validering, Nval)

        #biaspart = bias(test[:,2], zpredict_test, nr)
        #varians:

        #varpart = var2(zpredict_test, nr)

        print(varpart)
        varianceModel = varianceModel + varpart
        biasModel[i] = np.mean((y_test - (zpredict_test_validering))**2 )
        #print(biasModel[i])
        #plt.scatter(test[:,0], test[:,1], test[:,2], label='Data points')
        #plt.scatter(test[:,0], test[:,1], np.mean(zpredict_test, axis=1), label='Pred')
        #biaspart =
        #resetter:

        for a in range(0, k):
            if a == 0:
                zpredbias = zpredict_test
                ztruebias = test[:,2]
            else:
                zpredbias = np.concatenate((zpredbias, zpredict_test), axis = 0)
                ztruebias = np.concatenate((ztruebias, test[:,2]), axis = 0)
        varpart = 0
        betaCoeffpart = 0
        part_mse = 0
        nr = 0
        R_2 = 0
    #print(zpredbias.shape)
    #print(ztruebias.shape)
    #Nbias = np.size(ztruebias,0)

    # Beregnet mean squared error og R_2 etter cross-validation med k-fold :
    total_MSE = (mse/k)
    total_R_2 = (r2/k)
    #bias:
    #biasModelk = bias(ztruebias[:,0] , zpredbias[:,0] , Nbias)
    #biasModelk = bias(ztruebias, zpredbias, Nbias)

    #biasModelk = np.mean((ztruebias - np.mean(zpredbias))**2 )
    #print (biasModel)
    totVarianceModel = varianceModel/k
    #Intervall betacoefisienter:
    beta_confInt = np.c_[beta-(betaCoeff/k), beta+(betaCoeff/k)]
    return total_MSE, total_R_2, totVarianceModel , beta_confInt, biasModel


# In[19]:


def k_fold_Lasso(x,y,z, k, Lasso_alpha):
    # Oppretter en matrise av x, y og z. Datasettet som skal crossvalideres
    data = np.concatenate((x, y, z), axis = 1)

    n = np.size(data, 0) #rader
    m = np.size(data, 1) #kolonner

    # Shuffle dataset:
    np.random.shuffle(data)

    mse = 0
    r2 = 0

    #splitter dataset i k like deler. Oppretter antall folder. Bruker array_split. Da blir k deler blir ikke eksakt like,
    #men tilnærmet like, stort datasett gir lite forskjell.
    split_dataset = np.array_split(data,k)
    #Definerer alle k gangene datasettet skal deles opp.
    for i in range(0, k):
        # resetter train før hver oppdeling av datasettet.
        train = np.zeros((0,3))

        # j definerer selve splittdelene og man bestemmer her om del j er en trenings eller test del.
        for j in range(0,k):
            if j == i:
                test = split_dataset[j]
                continue
            train = np.concatenate((train, split_dataset[j]), axis = 0)
        print(train.shape)

        #Lasso_multi = np.c_[x_Lasso, y_Lasso]
        #Lasso_multi - Putter alt treningsettet i en multivariabel som sendes til sklearn sin Lassometode.
        Lasso_multi = np.c_[train[:,0], train[:,1]]
        Lasso_multi_test = np.c_[test[:,0],test[:,1]]

        poly2 = PolynomialFeatures(degree=10)

        # Transformerer variabelmatrisen
        Lasso_sklearn = poly2.fit_transform(Lasso_multi)
        Lasso_sklearn_test = poly2.fit_transform(Lasso_multi_test)

        #print(Lasso_sklearn)

        # Beregner Lasso modell
        lasso = linear_model.Lasso(alpha=Lasso_alpha)
        lasso.fit(Lasso_sklearn, train[:,2])

        # beregner forventet z-verdier av x og y verdier
        zpredict_test_L = lasso.predict(Lasso_sklearn_test)

        nrx =  np.size(test[:,0],0)
        nry =  np.size(test[:,1],0)
        nr =  np.size(test[:,2],0)

        # Finner forventet z_verdier for testsettet ved å bruke utregnet modell, beta, over og testsettets x og y verdier fra xyb-matrisa.


        # regner ut mean squared error pr fold
        part_mse = MSEk(test[:,2], zpredict_test_L, nr)
        mse = mse + part_mse

        z_mean = (1/nr ) * np.sum(test[:,2])

        # Regner ut R_2 score pr fold:
        R_2 = 1 - (((sum(test[:,2]-zpredict_test_L))**2)/(sum((test[:,2]-z_mean)**2)))
        r2 = r2 + R_2

        #x_d = test[:,0].reshape(nrx,nrx)
        #y_d =test[:,1].reshape(nry,nry)
        #z_d = zpredict_test_L.reshape(nr,nr)

        #plotSurface(x_d, y_d, z_d)

        part_mse = 0
        nr = 0
        R_2 = 0
    # Beregnet mean squared error og R_2 etter cross-validation med k-fold :
    total_MSE = (mse/k)
    total_R_2 = (r2/k)
    return total_MSE, total_R_2


# ## Part a)
# ### Linear Regression


#Henter meshgrid verdier av datasettet
x_d_OLS, y_d_OLS, z_d_OLS = FindData(X, Y)
print(x_d_OLS.shape)
# Plotter ekte overflate
print("DETTE ER OLS (KANSKJE)")
plotSurface(x_d_OLS, y_d_OLS, z_d_OLS)
exit(1)

#Konverterer meshgrideverdiene til kolonnevektorer
x_OLS,y_OLS,z_OLS,m_OLS = convertData(x_d_OLS, y_d_OLS, z_d_OLS)
print(m_OLS)
#print(x_OLS.shape)
# Finner designmatrisen for datasettet
xyb_Linear = XY(x_OLS,y_OLS)
# Finner modellen
betaLinear = beta_model('Linear', xyb_Linear, z_OLS)
#print(betaLinear.shape)


# ### Confidence Intervall med z_mean:
# Hvordan skal dette regnes ut og hva slags parametre skal velges. Brukes z_predict(confidenceIntervall) eller z_mean (confidenceIntervall2)?

# In[22]:


#print(confidenceIntervall2(z_OLS, m_OLS, 5, xyb_Linear, betaLinear ))


# ### Regner ut z_predict

# In[23]:


# regner ut forventet verdi ut ifra datasettet og forventet modell
zpredict_OLS, zpredict_OLS_mesh = predict(xyb_Linear,betaLinear,n)
print(zpredict_OLS.shape)

print(zpredict_OLS_mesh.shape)
print(n)


# Plotter forventet overflate
plotSurface(x_d_OLS, y_d_OLS, zpredict_OLS_mesh)


# ### Confidence Intervall med z_predict:

# In[25]:


#betaCoeff =
beta, betaCoeff, beta_confInt = confidenceIntervall(z_OLS, zpredict_OLS, m_OLS, 5, xyb_Linear, betaLinear )

print(betaCoeff)
print(beta)


# In[26]:


print(beta_confInt)


# In[27]:


# Regner ut gjennomsnittsverdien, variansen, mean squared error og standardavviket
z_mean_OLS = mu(z_OLS,m_OLS)
print(z_mean_OLS)
var_z_OLS = calc_Variance(z_OLS,z_mean_OLS,m_OLS)
print(var_z_OLS)
MSE_OLS = MSE(z_OLS, zpredict_OLS, m_OLS)
print(MSE_OLS)
R_2_OLS = calc_R_2(z_OLS, zpredict_OLS, z_mean_OLS, m_OLS)
print(R_2_OLS)


# $$E[(y - \hat f(x))^2] = \text{Bias}(\hat f(x))^2 + \text{Var}(\hat f(x)) + \sigma^2 \\ = \left(E[f(x) - \hat f(x)]^2\right)+ \left(E[\hat f(x)^2] - E[\hat f(x)]^2\right) + \sigma^2$$



print(m_OLS)
print(xyb_Linear.shape)
print(betaLinear.shape)


# ### Resampler settet en gang

# In[29]:


# #### Tester med nye random punkter:


# mse, R_2, biasModel, var, beta_confInt = test(x_OLS,y_OLS,z_OLS,'Linear')
# print('Mean squared error: ', mse)
# print('R 2- score: ', R_2)
# print('bias for the model ', biasModel)
# print('Variancen for the model: ', var)
# print('beta Confidence Intervall: \n', beta_confInt)

# In[30]:


mse, R_2, biasModel, var, beta_confInt = test2(x_OLS,y_OLS,z_OLS,'Linear')
print('Mean squared error: ', mse)
print('R 2- score: ', R_2)
print('bias for the model ', biasModel)
print('Variancen for the model: ', var)
print('beta Confidence Intervall: \n', beta_confInt)


# ### Tester med crossvalidation:

# In[31]:


total_MSE, total_R_2, totVarianceModel , beta_confInt, biasModelk = test_k_fold(x_OLS,y_OLS,z_OLS, 3, 'Linear')
print('Mean squared error: ', total_MSE)
print('R 2- score: ', total_R_2)
print('bias for the model ', biasModelk)
print('Variancen for the model: ', totVarianceModel)
print('beta Confidence Intervall: \n', beta_confInt)


# In[32]:


#numberofK = np.size(biasModelk,0)
biasTotal = np.mean(biasModelk) #/numberofK
print(biasTotal)


# In[33]:


print(z_OLS.shape)


# #### Finn bias og variansen for modellene.
#
# $$E[(y - \hat f(x))^2] = \text{Bias}(\hat f(x))^2 + \text{Var}(\hat f(x)) + \sigma^2 \\ = \left(E[f(x) - \hat f(x)]^2\right)+ \left(E[\hat f(x)^2] - E[\hat f(x)]^2\right) + \sigma^2$$
#
#
#
# mse = varians + bias

# In[34]:


bias2 = bias(z_OLS, zpredict_OLS, m_OLS)
print(bias2)
varianceModel = var2(zpredict_OLS, m_OLS)
print(varianceModel)


# ## Part b) Ridge
# #### Ridge Regression on the Franke function med resampling

# In[35]:


x_d_Ridge, y_d_Ridge, z_d_Ridge = FindData(X , Y)
print(x_d_Ridge.shape)
plotSurface(x_d_Ridge, y_d_Ridge, z_d_Ridge)


# In[36]:


x_Ridge, y_Ridge, z_Ridge, m = convertData(x_d_Ridge, y_d_Ridge, z_d_Ridge)
print(n)
print(m)
xyb_Ridge = XY(x_Ridge, y_Ridge)
betaRidge = beta_model('Ridge', xyb_Ridge, z_Ridge)


# In[37]:


zpredict_Ridge, zpredict_Ridge_mesh = predict(xyb_Ridge ,betaRidge, n)
print(zpredict_Ridge_mesh.shape)


# In[38]:


# Plot the predicted surface.
plotSurface(x_d_Ridge, y_d_Ridge, zpredict_Ridge_mesh)


# In[39]:


print(confidenceIntervall(z_Ridge, zpredict_Ridge, m, 5, xyb_Ridge, betaRidge ))


# In[40]:


# Regner ut gjennomsnittsverdien, variansen, mean squared error og standardavviket
z_mean_Ridge = mu(z_Ridge,m)
print(z_mean_Ridge)
#print(m)
var_z_Ridge = calc_Variance(z_Ridge,z_mean_Ridge,m)
print(var_z_Ridge)
MSE_Ridge = MSE(z_Ridge, zpredict_Ridge, m)
print(MSE_Ridge)
R_2_Ridge = calc_R_2(z_Ridge, zpredict_Ridge, z_mean_Ridge, m)
print(R_2_Ridge)


# In[41]:


mse_R, R_2_R, biasModel_R, var_R, beta_confInt_R = test(x_Ridge, y_Ridge, z_Ridge, 'Ridge')

print('Mean squared error: ', mse_R)
print('R 2- score: ', R_2_R)
print('bias for the model ', biasModel_R)
print('Variancen for the model: ', var_R)
print('beta Confidence Intervall: \n', beta_confInt_R)


# #### Endrer datasett til trening og test sett:

# In[42]:


total_MSE, total_R_2, totVarianceModel , beta_confInt, biasModel = test_k_fold(x_Ridge, y_Ridge, z_Ridge, 3, 'Ridge')


# In[43]:


print('Mean squared error: ', total_MSE)
print('R 2- score: ', total_R_2)
print('bias for the model ', biasModelk)
print('Variancen for the model: ', totVarianceModel)
print('beta Confidence Intervall: \n', beta_confInt)


# #### Finn bias og variansen for modellene.

# In[44]:


bias2 = bias(z_Ridge, zpredict_Ridge, m)
print(bias2)
varianceModel = var2(zpredict_Ridge, m)
print(varianceModel)


# ## Part c) Lasso

# In[45]:


# Oppretter meshgrid av datasettet
x_d_Lasso, y_d_Lasso, z_d_Lasso = FindData(X , Y)
print(x_d_Lasso.shape)
# Plotter z_d med meshgrid av x og y
plotSurface(x_d_Lasso, y_d_Lasso, z_d_Lasso)


# In[46]:


# Konverterer meshgrid til kolonnevektorer
x_Lasso, y_Lasso, z_Lasso, m = convertData(x_d_Lasso, y_d_Lasso, z_d_Lasso)
print(n)
print(m)
# Oppretter independent variabel matrise av x og y vektorene
Lasso_multi = np.c_[x_Lasso, y_Lasso]
#print (Lasso_multi)


# In[47]:


# Bruker sklearn for å finne Lasso-modell. Bruker PolynomialFeatures for ikke lineær funksjon
poly2 = PolynomialFeatures(degree=10)

# Transformerer variabelmatrisen
Lasso_sklearn = poly2.fit_transform(Lasso_multi)
print(Lasso_sklearn)
#print(z_Lasso)


# In[48]:


# Beregner Lasso modell
lasso=linear_model.Lasso(alpha=0.001)
lasso.fit(Lasso_sklearn, z_Lasso)

# beregner forventet z-verdier av x og y verdier
predictLasso = lasso.predict(Lasso_sklearn)

print("Lasso Coefficient: ", lasso.coef_)
print("Lasso Intercept: ", lasso.intercept_)


# In[49]:


# Plotter forventet overflate med Lasso modellen
predict_mesh_Lasso = predictLasso.reshape(n,n)
plotSurface(x_d_Lasso, y_d_Lasso, predict_mesh_Lasso)


# Kommentar til Lasso:
# Ser at ved å endre på alpha til veldig liten så begynner Lasso og ligne mer og mer på originale surfacen.
# ved alpha = 0.1 så synes bare et grått plan.
# Koeffisientene endres også ved å endre på alpha. Det samme med intercept.

# In[50]:


k_fold_Lasso(x_Lasso, y_Lasso, z_Lasso, 4, 0.001)






# ## Part d)
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('Data/SRTM_data_Norway_1.tif')
#terrain1 = imread('Data/SRTM_data_Norway_2.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Load the terrain
terrain2 = imread('Data/SRTM_data_Norway_2.tif')
# Show the terrain
plt.figure()
plt.title('Terrain over Norway 2')
plt.imshow(terrain2, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# ## Part e)

# In[53]:


# Velg ett område, f.eks 100x100 som tilsvarer X og Y. se på intensiteten på bildet.


# In[54]:


# Kode fra Kristine

import numpy as np
#from scipy.misc import imread
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def surface_plot(surface,title, surface1=None):
    M,N = surface.shape

    ax_rows = np.arange(M)
    ax_cols = np.arange(N)

    [X,Y] = np.meshgrid(ax_cols, ax_rows)

    fig = plt.figure()
    if surface1 is not None:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)

        ax = fig.add_subplot(1,2,2,projection='3d')
        ax.plot_surface(X,Y,surface1,cmap=cm.viridis,linewidth=0)
        plt.title(title)
    else:
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,surface,cmap=cm.viridis,linewidth=0)
        plt.title(title)


def predict(rows, cols, beta):
    out = np.zeros((np.size(rows), np.size(cols)))

    for i,y_ in enumerate(rows):
        for j,x_ in enumerate(cols):
            data_vec = np.array([1, x_, y_, x_**2, x_*y_, y_**2,                                 x_**3, x_**2*y_, x_*y_**2, y_**3,                                 x_**4, x_**3*y_, x_**2*y_**2, x_*y_**3,y_**4,                                 x_**5, x_**4*y_, x_**3*y_**2, x_**2*y_**3,x_*y_**4,y_**5])#,\
                            #    x_**6, x_**5*y_, x_**4*y_**2, x_**3*y_**3,x_**2*y_**4, x_*y_**5, y_**6, \
                            #    x_**7, x_**6*y_, x_**5*y_**2, x_**4*y_**3,x_**3*y_**4, x_**2*y_**5, x_*y_**6, y_**7, \
                            #    x_**8, x_**7*y_, x_**6*y_**2, x_**5*y_**3,x_**4*y_**4, x_**3*y_**5, x_**2*y_**6, x_*y_**7,y_**8, \
                            #    x_**9, x_**8*y_, x_**7*y_**2, x_**6*y_**3,x_**5*y_**4, x_**4*y_**5, x_**3*y_**6, x_**2*y_**7,x_*y_**8, y_**9])
            out[i,j] = data_vec @ beta

    return out

from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    # Load the terrain
    terrain1 = imread('data/SRTM_data_Norway_1.tif')
    [n,m] = terrain1.shape

    ## Find some random patches within the dataset and perform a fit

    #patch_size_row = 100
    #patch_size_col = 50

    patch_size_row = 800
    patch_size_col = 400

    # Define their axes
    rows = np.linspace(0,1,patch_size_row)
    cols = np.linspace(0,1,patch_size_col)

    [C,R] = np.meshgrid(cols,rows)

    x = C.reshape(-1,1)
    y = R.reshape(-1,1)

    num_data = patch_size_row*patch_size_col

    # Find the start indices of each patch

    num_patches = 5

    np.random.seed(4155)

    row_starts = np.random.randint(0,n-patch_size_row,num_patches)
    col_starts = np.random.randint(0,m-patch_size_col,num_patches)

    for i,row_start, col_start in zip(np.arange(num_patches),row_starts, col_starts):
        row_end = row_start + patch_size_row
        col_end = col_start + patch_size_col

        patch = terrain1[row_start:row_end, col_start:col_end]

        z = patch.reshape(-1,1)

        # Perform OLS fit
        data = np.c_[np.ones((num_data,1)), x, y,                      x**2, x*y, y**2,                      x**3, x**2*y, x*y**2, y**3,                      x**4, x**3*y, x**2*y**2, x*y**3,y**4,                      x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]#, \
                     #x**6, x**5*y, x**4*y**2, x**3*y**3,x**2*y**4, x*y**5, y**6, \
                     #x**7, x**6*y, x**5*y**2, x**4*y**3,x**3*y**4, x**2*y**5, x*y**6, y**7, \
                     #x**8, x**7*y, x**6*y**2, x**5*y**3,x**4*y**4, x**3*y**5, x**2*y**6, x*y**7,y**8, \
                     #x**9, x**8*y, x**7*y**2, x**6*y**3,x**5*y**4, x**4*y**5, x**3*y**6, x**2*y**7,x*y**8, y**9]

        beta_ols = np.linalg.inv(data.T @ data) @ data.T @ z

        fitted_patch = predict(rows, cols, beta_ols)

        mse = np.sum( (fitted_patch - patch)**2 )/num_data
        R2 = 1 - np.sum( (fitted_patch - patch)**2 )/np.sum( (patch - np.mean(patch))**2 )
        var = np.sum( (fitted_patch - np.mean(fitted_patch))**2 )/num_data
        bias = np.sum( (patch - np.mean(fitted_patch))**2 )/num_data

        print("patch %d, from (%d, %d) to (%d, %d)"%(i+1, row_start, col_start, row_end,col_end))
        print("mse: %g\nR2: %g"%(mse, R2))
        print("variance: %g"%var)
        print("bias: %g\n"%bias)

        surface_plot(fitted_patch,'Fitted terrain surface',patch)

    # Perform fit over the whole dataset
    print("The whole dataset")

    rows = np.linspace(0,1,n)
    cols = np.linspace(0,1,m)

    [C,R] = np.meshgrid(cols,rows)

    x = C.reshape(-1,1)
    y = R.reshape(-1,1)

    num_data = n*m

    data = np.c_[np.ones((num_data,1)), x, y,                  x**2, x*y, y**2,                  x**3, x**2*y, x*y**2, y**3,                  x**4, x**3*y, x**2*y**2, x*y**3,y**4,                  x**5, x**4*y, x**3*y**2, x**2*y**3,x*y**4, y**5]

    z = terrain1.flatten()

    beta_ols = np.linalg.inv(data.T @ data) @ data.T @ z

    fitted_terrain = predict(rows, cols, beta_ols)

    mse = np.sum( (fitted_terrain - terrain1)**2 )/num_data
    R2 = 1 - np.sum( (fitted_terrain - terrain1)**2 )/np.sum( (terrain1- np.mean(terrain1))**2 )
    var = np.sum( (fitted_terrain - np.mean(fitted_terrain))**2 )/num_data
    bias = np.sum( (terrain1 - np.mean(fitted_terrain))**2 )/num_data

    print("mse: %g\nR2: %g"%(mse, R2))
    print("variance: %g"%var)
    print("bias: %g\n"%bias)

    plt.show()
