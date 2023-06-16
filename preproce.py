import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as stg

def preprocessing(file_path):
    df = pd.read_csv(file_path)
    
    # Convertir la colonne de date en format de date
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    return df

def main():
    df = preprocessing("C:/Users/sylva/AppData/Local/GitHubDesktop/app-3.2.0/Projet_ARIMA_2023/covid_CIV_dataset_projet.csv")

    
    # Affichage de la donnée
    print(df)
    print("\n")
    
    # # Suppression des colonnes 
    df = df.drop(['location'], axis=1)
    
    print(df)
    print("\n")
  
    # Vérification des éléments manquants
    print(df.isnull().sum())
    
    # Remplacer les données manquantes par interpolation
    df = df.interpolate()
    
    # Vérification et nettoyage des données
    print(df.isnull().sum())
    print("\n")
    print(df)
    print("\n")
    
    # Faire une copie de la dataframe
    df1 = df.copy()
    print(df1)
    print("\n")
    
    # Convertir l'index en date compréhensible
    df1['date'] = pd.to_datetime(df1['date'], format='%d/%m/%Y')
    df1.index = df1['date']
    
    # Affichage des statistiques descriptives
    print(df1.describe())
    

    print(df1)
    print("\n")
   
    # Vérification de la tendance (Graphique Histogramme)
    sns.distplot(df1['total_cases'], hist=False)
    plt.show()
    
    # La Forme du modèle
    print("Additif".center(50, "-"))
    MDA = seasonal_decompose(df1['total_cases'], period=12).plot()
    print("Multiplicatif".center(50, "-"))
    MDM = seasonal_decompose(df1['total_cases'], model='multiplicative', period=12).plot()
    
    #   La variance est trop grande ce qui nous donne un modèle multiplicatif.
    
    df1['total_cases'] = np.log(df1['total_cases'])
    print(df1)
    
    # La Vérification
    sns.distplot(df1['total_cases'], hist=False)
    plt.show()
    
    # Une seconde approche est: la différenciation
    df1 = df1.diff().dropna()
    print(df1)
    
    # Vérification de la distribution
    sns.distplot(df1['total_cases'], hist=False)
    plt.show()
    
    # ACF (Autocorrelation)
    resultat_adf1 = adfuller(df1['total_cases'])
    print(resultat_adf1)
    ACF = resultat_adf1[0]
    print(ACF)
    p_value = resultat_adf1[1]
    print(p_value)
    
    result = seasonal_decompose(df1['total_cases'], model='additive', period=4)
    result.plot()
    plt.show()
    

    
    # Courbe d'autocorrélation et autocorrélation partielle
    stg.plot_acf(df1['total_cases'])
    plt.show()
    stg.plot_pacf(df1['total_cases'])
    plt.show()
    
    return df1
