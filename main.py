from prepro import preprocessing
from prepare import prepare_data
from lire import ARIMAModel

def main():
    
    df = preprocessing("C:/Users/sylva/AppData/Local/GitHubDesktop/app-3.2.0/Projet_ARIMA/consommation-quotidienne-brute.csv")
    t_train, t_test ,t_validation, y_train, y_test, y_validation=prepare_data(df, 0.6, 0.2)
    
    print(t_train)
    print(t_test)
    print(t_validation)
    
    print(y_train)
    print(y_test)
    print(y_validation)
    model = ARIMAModel(12,1,12,t_train, t_test ,t_validation, y_train, y_test, y_validation)
    model.training()
    model.show_forcast_of_arima_model()

    
    
    
if __name__=="__main__": 
        
        main()
        
        
    