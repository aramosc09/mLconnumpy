import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import json
import os

def get_fetal_dataset(file_path,corr_threshold):
    # Import csv
    data = pd.read_csv(file_path)
    print('------------------------------------------------------------------')
    print('Columns with high correlation for target column (fetal_health)')
    print('------------------------------------------------------------------')
    print(data.corr()['fetal_health'][abs(data.corr()['fetal_health']) >= corr_threshold])
    print('------------------------------------------------------------------')
    cols = list(data.corr()['fetal_health'][abs(data.corr()['fetal_health']) >= corr_threshold].index)
    cols.remove('fetal_health')
    data['fetal_health'] = data['fetal_health'].apply(lambda x: 0 if x == 1 else 1)

    # Separar las características (X) de la variable objetivo (y)
    X = data[cols]
    y = data['fetal_health']

    # Cargar el archivo JSON y convertirlo en un diccionario
    with open('/Users/ajelandro/Documents/GitHub/mLconnumpy/data_description.json', 'r') as archivo:
        data_description = json.load(archivo)

    description_flag = input('View data description? (y/n)      ')

    if description_flag == 'y':
        for col in cols:
            print('\033[1m\033[38;5;214m' + col + '\033[0m' + ':', data_description[col])
        print('------------------------------------------------------------------')
        print('Proceeding to get distribution plots...')
    elif description_flag == 'n':
        print('------------------------------------------------------------------')
        print('Proceeding to get distribution plots...')
    else:
        print('------------------------------------------------------------------')
        print('No valid answer received. Proceeding to get distribution plots...')

    return data[cols], X, y, data_description

def get_distribution_plots(data,cols,data_description):
    for col in cols:
        plt.figure(figsize=(6, 6))
        sns.histplot(data[col])
        plt.xlabel(data_description[col])
        plt.savefig('distribution_'+col+'.png')
    print('Generated Distribution plots for:',str(cols))
    print('------------------------------------------------------------------')

# Funciones auxiliares para la regresión logística
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = X.shape[0]
    h = sigmoid(np.dot(X, weights))
    cost = (-1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = X.shape[0]
    for i in range(iterations):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
    return weights

def train_rl(X, y, learning_rate, iterations, data_description, sigmoid_input):
    print('Training LR...')
    y = (y == 1).astype(int)

    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Inicialización de los pesos
    weights = np.zeros(X_train.shape[1])

    # Entrenar el modelo utilizando gradiente descendente
    weights = gradient_descent(X_train, y_train, weights, learning_rate, iterations)

    # Predicción y evaluación en el conjunto de prueba
    y_pred = sigmoid(np.dot(X_test, weights)) >= sigmoid_input
    accuracy = np.mean(y_pred == y_test)
    print('\033[1m\033[32mTraining complete!\033[0m')
    print('------------------------------------------------------------------')

    print('Accuracy:', accuracy)
    print('------------------------------------------------------------------')

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Visualizar la matriz de confusión usando seaborn
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    used_cols = []
    for col in list(X.columns):
        used_cols.append(data_description[col])

    plt.title('RL usando: ' + str(used_cols).replace('[','').replace(']',''))
    plt.ylabel('Actual')
    plt.xlabel('Predicción')
    plt.savefig('confusion_matrix_' + 'learning_rate=' + str(learning_rate) + '_iterations=' + str(iterations) + '.png')

    return y_test, y_pred, accuracy, weights, scaler

def predict(X, weights, scaler):
    print('Please enter data for model to predict:')

    new_data_values = []
    for col in list(X.columns):
        instance = input('      '+'\033[1m\033[38;5;214m' + col + '\033[0m'+': ')
        new_data_values.append(float(instance))  # Asegúrate de que sean valores numéricos

    new_data = pd.DataFrame([new_data_values], columns=list(X.columns))

    # Normalizar las características del nuevo conjunto de datos utilizando el scaler ajustado
    new_data_scaled = scaler.transform(new_data)
    # Realizar la predicción
    new_prediction = sigmoid(np.dot(new_data_scaled, weights)) >= 0.5
    print('------------------------------------------------------------------')

    if new_prediction[0] == False:
        print("\033[1mPrediction\033[0m: Fetal health is normal.")
        print('------------------------------------------------------------------')
    else:
        print("\033[1mPrediction\033[0m: Suspect fetal health detected. Further study is highly recommended.")
        print('------------------------------------------------------------------')

def main():
    file_path = "/Users/ajelandro/Documents/GitHub/mLconnumpy/fetal_health.csv"
    input_data_path = "/Users/ajelandro/Documents/GitHub/mLconnumpy/input_data.json"
    with open(input_data_path, 'r') as archivo:
        input_data = json.load(archivo)

    print(input_data)
    data, X, y, data_description = get_fetal_dataset(file_path, input_data["correlation"])
    get_distribution_plots(data, list(X.columns), data_description)
    y_test, y_pred, accuracy, weights, scaler = train_rl(X, y, input_data["learning_rate"], input_data["iterations"], data_description, input_data['sigmoid'])
    predict(X, weights, scaler)

if __name__ == '__main__':
    main()