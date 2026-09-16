# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:11:01 2025

@author: u0150711
"""

###Code for filling gaps using Random forest and doble mass curve
##In principle, stations with less than a year of gaps, will be filled with ML,
##else they will be filled with the nearest station

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

path=r'C:\Workdir\PhD_data\08_Datos\Procesados'
os.chdir(path)

data=pd.read_excel('Matriz_meteo.xlsx',skiprows=[1,2,3,4],index_col=0)

estaciones=pd.read_excel('Matriz_meteo.xlsx',index_col=0).iloc[:3]
estaciones=estaciones.T
estaciones['Estacion']=estaciones.index

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance (in km) between two points using the Haversine formula."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in km

def compute_3d_distance(lat1, lon1, elev1, lat2, lon2, elev2):
    """Compute 3D distance incorporating elevation differences."""
    dist_2d = haversine(lat1, lon1, lat2, lon2)  # Horizontal distance in km
    elev_diff = (elev2 - elev1) / 1000  # Convert elevation difference to km
    return np.sqrt(dist_2d ** 2 + elev_diff ** 2)  # 3D Euclidean distance

def find_nearest_stations(df):
    """
    Find the nearest station for each weather station in the dataset.

    Parameters:
    df (pd.DataFrame): DataFrame with 'Latitud', 'Longitud', and 'Altitud' columns.

    Returns:
    pd.DataFrame: A DataFrame with the nearest station for each location.
    """
    num_stations = len(df)
    nearest_stations = []

    for i in range(num_stations):
        min_dist = float('inf')
        nearest_index = None

        for j in range(num_stations):
            if i == j:
                continue  # Skip self-comparison

            dist = compute_3d_distance(df.iloc[i]["Latitud"], df.iloc[i]["Longitud"], df.iloc[i]["Altitud"],
                                       df.iloc[j]["Latitud"], df.iloc[j]["Longitud"], df.iloc[j]["Altitud"])
            if dist < min_dist:
                min_dist = dist
                nearest_index = j

        nearest_stations.append((df.iloc[i]["Estacion"], df.iloc[nearest_index]["Estacion"], min_dist))

    return pd.DataFrame(nearest_stations, columns=["Estacion", "Nearest_Station", "Distance_km"])

nearest_df = find_nearest_stations(estaciones)


def filter_date(df):
    start_date='1970-01-01'
    end_date='2023-12-31'
    mask = (df.index >= start_date) & (df.index <= end_date)
    df=df.loc[mask]
    return df

data_filtered=filter_date(data)

train_data=data_filtered.loc['1970':'2014']
test_data=data_filtered.loc['2016':'2023']

validation_data=data_filtered.loc['01-08-2015':'31-07-2016']

data_precip=data.loc['1970':'2023']

nan_count = train_data.isna()
print("vacio",nan_count.sum())
nan_count2 = test_data.isna()
print("vacio",nan_count2.sum())

##Method 1: Curva doble masa

def acumulatives(df, referencia, relleno):
    if referencia not in df.columns or relleno not in df.columns:
        print(f"One of the columns '{referencia}' or '{relleno}' does not exist in the DataFrame.")
        return
    
    # Drop rows with NaN values in the specified columns
    df = df.dropna(subset=[referencia, relleno])
    dt = df[referencia].copy()
    df2 = df[relleno].copy()
    
    dt['acumulada'] = dt.cumsum()
    df2['acumulada'] = df2.cumsum()
    
    slope, intercept, r_value, p_value, std_err = linregress(dt['acumulada'], df2['acumulada'])
    plt.scatter(dt['acumulada'], df2['acumulada'])
    plt.plot(dt['acumulada'], slope * dt['acumulada'] + intercept, 'b-', label=f'Linear Fit (R² = {r_value**2:.4f})')
    equation_text = f"y = {slope:.4f}x + {intercept:.2f}\nR² = {r_value**2:.4f}"
    plt.text(0.05, 0.85, equation_text, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel("Pcp referencia "+referencia)
    plt.ylabel("Pcp relleno "+relleno)
    plt.legend()
    plt.show()
    print (round(r_value**2,5))
    return slope


slope=acumulatives(train_data, 'Colomi', 'Embalse Corani')

slope=acumulatives(train_data, 'Cochabamba Aeropuerto', 'Sarco')

slope=acumulatives(train_data, 'Arani', 'Tiraque')

def relleno_climatologico2(df_referencia,columns_to_fill):
    #df_referencia= period of data which is used as basis, and the period in which data is filled
        
    relleno_climatologico=df_referencia.copy()
    relleno_climatologico=relleno_climatologico.dropna()
    relleno_climatologico['DOY']=relleno_climatologico.index.dayofyear
    doy_mean=round(relleno_climatologico.groupby('DOY').mean(),3)
    
    # Define columns to fill
    # Add DOY column to train_data
    df_referencia['DOY'] = df_referencia.index.dayofyear

    # Fill missing values using the climatological mean for each DOY
    for col in columns_to_fill:
        df_referencia[col] = df_referencia.apply(lambda row: doy_mean.at[row['DOY'], col] if pd.isna(row[col]) else row[col], axis=1)

    # Drop DOY column (optional)
    df_referencia.drop(columns=['DOY'], inplace=True)

    nan_count = df_referencia.isna()
    print("vacio2",nan_count.sum())
    
    df_relleno=df_referencia.copy()
    return df_relleno

def relleno_climatologico(df_referencia, columns_to_fill):
    # Make a copy of the dataframe
    df_referencia = df_referencia.copy()

    # Separate data into two periods
    df_train = df_referencia[df_referencia.index.year <= 2015]  # 1970-2014 (to be filled)
    df_future = df_referencia[df_referencia.index.year > 2015]  # 2015-2023 (unchanged)

    # Remove NaNs from training data and compute climatological mean
    df_train_clean = df_train.dropna().copy()
    df_train_clean['DOY'] = df_train_clean.index.dayofyear
    doy_mean = df_train_clean.groupby('DOY').mean().round(3)

    # Add DOY column to training data
    df_train['DOY'] = df_train.index.dayofyear

    # Fill missing values in the 1970-2014 period
    for col in columns_to_fill:
        df_train[col] = df_train.apply(
            lambda row: doy_mean.at[row['DOY'], col] if pd.isna(row[col]) else row[col],
            axis=1
        )

    # Drop DOY column
    df_train.drop(columns=['DOY'], inplace=True)

    # Merge the two periods back together
    df_relleno = pd.concat([df_train, df_future]).sort_index()

    # Count remaining NaNs
    print("Valores faltantes después del relleno:\n", df_relleno.isna().sum())

    return df_relleno

def relleno_doblemasa(df,referencia,relleno,valor):
    x=valor
    prueba_relleno=df[[referencia, relleno]].copy()
    prueba_relleno['Rellenado']=0
    prueba_relleno['Rellenado'] = prueba_relleno[relleno].fillna(prueba_relleno[referencia] * x) 
    
    nan_count = prueba_relleno['Rellenado'].isna()
    print("vacio",nan_count.sum())
    
    plt.figure(figsize=(12,6))
    plt.plot(prueba_relleno['Rellenado'],label='Rellenado')
    plt.plot(prueba_relleno[relleno],label='Original data')
    plt.legend()
    plt.xlim([16500,19723])
    plt.show()
    
    df_rellenado=prueba_relleno['Rellenado'].copy().to_frame()
    df_rellenado=df_rellenado.rename(columns={'Rellenado':relleno})
    return df_rellenado

#%%

def relleno_random_forest1(df, relleno):
    #get a funtion that fits the train data
    ## Random Forest fill gaps using a sliding window approach
    train_data_est1=df[relleno].copy().to_frame()
   
    # Identify missing values
    train_data_est1[relleno] = train_data_est1[relleno].replace(-9999, np.nan)

    def create_features_targets(df, label, window=60):
        ##window defines the past days of precipitation to forecast
        ##In this case is defined by 60 days to forecast the next days
        x, y ,indices= [], [],[]
        for i in range(window, len(df)):
            if not np.isnan(df.iloc[i][label]):  # Only train on known values
                x.append(df.iloc[i-window:i][label].values)
                y.append(df.iloc[i][label])
                indices.append(df.index[i])
        return np.array(x), np.array(y),indices

    # Train-test split
    df_train = train_data_est1.loc["1970-01-01":"2015-12-31"]
    df_test = train_data_est1.loc["2015-01-01":"2023-12-31"]
    df_val = train_data_est1.loc["2014-08-01":"2016-07-31"]

    # Prepare data for models
    X_train, y_train, _ = create_features_targets(df_train,relleno)
    X_test, _ ,gap_indices= create_features_targets(df_test,relleno)
    X_val, y_val,val_indices = create_features_targets(df_val,relleno)
    # RANDOM FOREST MODEL
    rf = RandomForestRegressor(n_estimators=1000, random_state=50,criterion='squared_error')
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)

    rf_preds2 = rf.predict(X_train)

    rf_preds3 = rf.predict(X_val)

    rmse_window = np.sqrt(mean_squared_error(y_val, rf_preds3))
    print(f"Validation Window RMSE: {rmse_window:.3f} mm")


    # Store results in the original dataframe
    train_data_est1.loc[df_test.index[-len(rf_preds):], "rf_filled"] = rf_preds
    train_data_est1.loc[df_val.index[-len(rf_preds3):], "rf_filled3"] = rf_preds3
    train_data_est1.loc[df_train.index[-len(rf_preds2):], "rf_filled2"] = rf_preds2

    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(train_data_est1[relleno], label="Original Data")
    plt.plot(train_data_est1["rf_filled"], label="Random Forest Prediction")
    plt.xlim([16500,19723])
    plt.title('Funcion Random Forest - Time Window slide'+relleno)
    plt.legend()
    plt.show()
  
    train_data_est1["filled_precipitation"] = train_data_est1[relleno].fillna(train_data_est1["rf_filled"])
    
    df_rellenado=train_data_est1["filled_precipitation"].copy().to_frame()
    df_rellenado=df_rellenado.rename(columns={'filled_precipitation':relleno})

    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(df_rellenado[relleno], label="Rellenado")
    plt.plot(train_data_est1[relleno], label="Original Data")
    plt.xlim([16500,19723])## change according to period with gaps
    plt.title('Comparacion de relleno'+relleno)
    plt.legend()
    plt.show()

    return df_rellenado

def relleno_random_forest2(df, relleno):
    #get a funtion that fits the train data
    ## Random Forest fill gaps considering Time based approach
    df_data_est1=data_filtered[relleno].copy().to_frame()

    # Create time-based features
    df_data_est1["year"] = df_data_est1.index.year
    df_data_est1["month"] = df_data_est1.index.month
    df_data_est1["day_of_year"] = df_data_est1.index.dayofyear

    # Split dataset into training, validation, and filling periods
    train_df = df_data_est1.loc["1970-01-01":"2015-12-31"].dropna()
    validation_df = df_data_est1.loc["2014-08-01":"2016-07-31"].dropna()
    gap_fill_df = df_data_est1.loc["2016-01-01":"2023-12-31"]

    # Prepare training data (X = time features, y = precipitation)
    X_train = train_df[["year", "month", "day_of_year"]]
    y_train = train_df[relleno]

    X_val = validation_df[["year", "month", "day_of_year"]]
    y_val = validation_df[relleno]

    X_gap = gap_fill_df[["year", "month", "day_of_year"]]

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=1000, random_state=50,criterion='squared_error')
    rf.fit(X_train, y_train)    

    # Validate Model (2014-2016)
    val_preds = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation TBA RMSE: {rmse:.3f} mm")


    # Fill Missing Values (2016-2023)
    gap_preds = rf.predict(X_gap)
    df_data_est1.loc[X_gap.index, "rf_filled"] = gap_preds

    # Fill NaN values in the original column
    df_data_est1["filled_precipitation"] = df_data_est1[relleno].fillna(df_data_est1["rf_filled"])

    plt.figure(figsize=(12,6))
    plt.plot(df_data_est1[relleno], label="Original Data")
    plt.plot(df_data_est1["rf_filled"], label="Random Forest Prediction")
    plt.title('Funcion Random Forest - Time Based Approach'+relleno)
    plt.xlim([16500,19723])
    plt.legend()
    plt.show()

    df_rellenado=df_data_est1["filled_precipitation"].copy().to_frame()
    df_rellenado=df_rellenado.rename(columns={'filled_precipitation':relleno})
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(df_rellenado[relleno], label="Rellenado")
    plt.plot(df_data_est1[relleno], label="Original Data")
    plt.xlim([16500,19723])## change according to period with gaps
    plt.title('Comparacion de relleno'+str(relleno))
    plt.legend()
    plt.show()

    return df_rellenado
#%%
#relleno_rf1=relleno_random_forest1(data_filtered, 'Cochabamba Aeropuerto')
relleno_rf2=relleno_random_forest2(data_filtered, 'Cochabamba Aeropuerto')

#relleno historico 1970-2015 estaciones 
#'Sacaba', 'Santivañez', 'Sivingani-Mizque', 'Morochata'
# Define columns to fill
columns_to_fill = ['Sacaba', 'Santivañez', 'Sivingani-Mizque', 'Morochata']

data_eval=relleno_climatologico(data_filtered, columns_to_fill)

data_eval1=data_eval.loc["1970-01-01":"2015-12-31"]
nan_count = data_eval1.isna()
print("vacio",nan_count.sum())

data_eval2=data_eval.loc['2016':'2023']
nan_count = data_eval2.isna()
print("vacio",nan_count.sum())

#%% Aplication of the data filling methods
#anzaldo_filled_rf1=relleno_random_forest1(data_filtered, 'Anzaldo')
anzaldo_filled_rf2=relleno_random_forest2(data_filtered, 'Anzaldo')

#Colomi_filled_rf1=relleno_random_forest1(data_eval, 'Colomi')
Colomi_filled_rf2=relleno_random_forest2(data_eval, 'Colomi')

#pairumani_filled_rf1=relleno_random_forest1(data_eval, 'Pairumani')
pairumani_filled_rf2=relleno_random_forest2(data_eval, 'Pairumani')

#sbenito_filled_rf1=relleno_random_forest1(data_eval, 'San Benito')
sbenito_filled_rf2=relleno_random_forest2(data_eval, 'San Benito')

#santivanez_filled_rf1=relleno_random_forest1(data_eval, 'Santivañez')
est_santivanez=data_eval['Santivañez'].copy().to_frame()
santivanez_filled_rf2=relleno_random_forest2(est_santivanez, 'Santivañez')
santivanez_filled_rf2['Santivañez']=santivanez_filled_rf2['Santivañez'].fillna(est_santivanez['Santivañez'])

#parotani_filled_rf1=relleno_random_forest1(data_eval, 'Parotani')
parotani_filled_rf2=relleno_random_forest2(data_eval, 'Parotani')

#arani_filled_rf1=relleno_random_forest1(data_eval, 'Arani')
arani_filled_rf2=relleno_random_forest2(data_eval, 'Arani')

#%%filled with doble masa or random forest
#make dataframe wit referencia and relleno stations
slope=acumulatives(train_data, 'Colomi', 'Embalse Corani')

slope=acumulatives(train_data, 'Cochabamba Aeropuerto', 'Sarco')

slope=acumulatives(train_data, 'Arani', 'Tiraque')

slope=acumulatives(train_data, 'San Benito', 'Tarata')

slope=acumulatives(train_data, 'Sarco', 'La Violeta')

slope=acumulatives(train_data, 'Cochabamba Aeropuerto', 'La Violeta')

slope=acumulatives(train_data, 'Anzaldo','Sacabamba')

slope=acumulatives(train_data, 'Templo','Morochata')

slope=acumulatives(train_data, 'Cuatro Esquinas Misicuni','Caluyo')

slope=acumulatives(train_data, 'Sacabamba','Sivingani-Mizque')

'Sacabamba','Sivingani-Mizque'

###pair 1: Colomi-Embalse Corani
df_colomi_corani=Colomi_filled_rf2['Colomi'].to_frame()
df_colomi_corani['Embalse Corani']=data_eval['Embalse Corani'].copy()

#test functions
corani_filled_rf2=relleno_random_forest2(data_eval, 'Embalse Corani')
corani_filled2=relleno_doblemasa(df_colomi_corani,'Colomi','Embalse Corani',4.2173)

##### cbba-sarco
df_cbba_sarco=relleno_rf2['Cochabamba Aeropuerto'].to_frame()
df_cbba_sarco['Sarco']=data_eval['Sarco'].copy()

sarco_filled_rf2=relleno_random_forest2(data_eval, 'Sarco')
sarco_filled2=relleno_doblemasa(df_cbba_sarco,'Cochabamba Aeropuerto','Sarco',1.0461)

##### arani-tiraque
df_arani_tiraque=arani_filled_rf2['Arani'].to_frame()
df_arani_tiraque['Tiraque']=data_eval['Tiraque'].copy()

tiraque_filled_rf2=relleno_random_forest2(data_eval, 'Tiraque')
tiraque_filled2=relleno_doblemasa(df_arani_tiraque,'Arani','Tiraque',1.4590)

#### San Benito -  Tarata
df_sbenito_tarata=sbenito_filled_rf2['San Benito'].to_frame()
df_sbenito_tarata['Tarata']=data_eval['Tarata'].copy()

tarata_filled_rf2=relleno_random_forest2(data_eval,'Tarata')
tarata_filled2=relleno_doblemasa(df_sbenito_tarata,'San Benito','Tarata',1.6911)

#### Sarco -  La violeta
df_sarco_violeta=sarco_filled2['Sarco'].to_frame()
df_sarco_violeta['La Violeta']=data_eval['La Violeta'].copy()

lavioleta_filled_rf2=relleno_random_forest2(data_eval,'La Violeta')
lavioleta_filled2=relleno_doblemasa(df_sarco_violeta,'Sarco','La Violeta',1.1349)

#### Sacabamba -  Anzaldo
df_sacabamba_anzaldo=anzaldo_filled_rf2['Anzaldo'].to_frame()
df_sacabamba_anzaldo['Sacabamba']=data_eval['Sacabamba'].copy()

sacabamba_filled_rf2=relleno_random_forest2(data_eval, 'Sacabamba')
sacabamba_filled2=relleno_doblemasa(df_sacabamba_anzaldo,'Anzaldo','Sacabamba',1.2533)

#### Morochata - Templo
df_morochata_templo=data_eval['Templo'].to_frame()
df_morochata_templo['Morochata']=data_eval['Morochata'].copy()

morochata_filled_rf2=relleno_random_forest2(data_eval,'Morochata')
morochata_filled2=relleno_doblemasa(df_morochata_templo,'Templo','Morochata',0.9287)

#### Caluyo - Cuatro Esquinas
df_caluyo_esquinas=data_eval['Cuatro Esquinas Misicuni'].to_frame()
df_caluyo_esquinas['Caluyo']=data_eval['Caluyo'].copy()

caluyo_filled_rf2=relleno_random_forest2(data_eval, 'Caluyo')
caluyo_filled=relleno_doblemasa(df_caluyo_esquinas,'Cuatro Esquinas Misicuni','Caluyo',0.7056)

#### Sivingani Mizque - Sacabamba
df_sivingani_sacabamba=sacabamba_filled2['Sacabamba'].to_frame()
df_sivingani_sacabamba['Sivingani-Mizque']=data_eval['Sivingani-Mizque'].copy()

sivingani_filled_rf2=relleno_random_forest2(data_eval,'Sivingani-Mizque')
sivingani_filled2=relleno_doblemasa(df_sivingani_sacabamba,'Sacabamba','Sivingani-Mizque',0.9821)

#%%Exportar los datos

salida=r'C:\Workdir\PhD_data\08_Datos\Procesados'
#train_data_est1['Filled'].to_excel('Cochabamba Aeropuerto Rellenado 2023.xlsx')
##Create a single data frame with all the fillgaps

estaciones_rellenadas=anzaldo_filled_rf2.copy()
estaciones_rellenadas['Arani']=arani_filled_rf2['Arani'].copy()
estaciones_rellenadas['Cochabamba Aeropuerto']=relleno_rf2['Cochabamba Aeropuerto'].copy()
estaciones_rellenadas['Colomi']=Colomi_filled_rf2['Colomi'].copy()
estaciones_rellenadas['Cuatro Esquinas Misicuni']=data_eval['Cuatro Esquinas Misicuni'].copy()
estaciones_rellenadas['La Violeta']=lavioleta_filled2['La Violeta'].copy()
estaciones_rellenadas['Pairumani']=pairumani_filled_rf2['Pairumani'].copy()
estaciones_rellenadas['Parotani']=parotani_filled_rf2['Parotani'].copy()
estaciones_rellenadas['Sacaba']=data_eval['Sacaba'].copy()
estaciones_rellenadas['Sacabamba']=sacabamba_filled2['Sacabamba'].copy()
estaciones_rellenadas['San Benito']=sbenito_filled_rf2['San Benito'].copy()
estaciones_rellenadas['Santivañez']=santivanez_filled_rf2['Santivañez'].copy()
estaciones_rellenadas['Sarco']=sarco_filled2['Sarco'].copy()
estaciones_rellenadas['Sivingani-Mizque']=sivingani_filled2['Sivingani-Mizque'].copy()
estaciones_rellenadas['Tarata']=tarata_filled2['Tarata'].copy()
estaciones_rellenadas['Tiraque']=tiraque_filled2['Tiraque'].copy()
estaciones_rellenadas['Sunjani']=data_eval['Sunjani'].copy()
estaciones_rellenadas['Templo']=data_eval['Templo'].copy()
estaciones_rellenadas['Morochata']=morochata_filled2['Morochata'].copy()
estaciones_rellenadas['Embalse Corani']=corani_filled2['Embalse Corani'].copy()
estaciones_rellenadas['Caluyo']=caluyo_filled['Caluyo'].copy()


##verificacion vacios
nan_filled = estaciones_rellenadas.isna()
print("vacio:",nan_filled.sum())

#estaciones_rellenadas.to_excel('Matriz_precipitacion_rellenada_1970_2023.xlsx')

