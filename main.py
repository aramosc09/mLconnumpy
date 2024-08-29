import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

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
    return data[cols], X, y

def get_distribution_plots(data,cols):
    for col in cols:
        plt.figure(figsize=(6, 6))
        sns.histplot(data[col])
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

def train_rl(X, y, learning_rate, iterations):
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
    y_pred = sigmoid(np.dot(X_test, weights)) >= 0.5
    accuracy = np.mean(y_pred == y_test)
    print('Training complete!')
    print('------------------------------------------------------------------')

    print('Accuracy:', accuracy)
    print('------------------------------------------------------------------')

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Visualizar la matriz de confusión usando seaborn
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    plt.title('RL usando ' + str(list(X.columns)))
    plt.ylabel('Actual')
    plt.xlabel('Predicción')
    plt.savefig('confusion_matrix_' + 'learning_rate=' + str(learning_rate) + '_iterations=' + str(iterations) + '.png')

    return y_test, y_pred, accuracy, weights, scaler

def predict(X, weights, scaler):
    print('Predict...')

    new_data_values = []
    for col in list(X.columns):
        instance = input(col+': ')
        new_data_values.append(float(instance))  # Asegúrate de que sean valores numéricos

    new_data = pd.DataFrame([new_data_values], columns=list(X.columns))

    # Normalizar las características del nuevo conjunto de datos utilizando el scaler ajustado
    new_data_scaled = scaler.transform(new_data)

    # Realizar la predicción
    new_prediction = sigmoid(np.dot(new_data_scaled, weights)) >= 0.5
    if new_prediction[0] == False:
        print("Predicción: El feto es sano.")
    else:
        print("Predicción: La salud del feto puede estar en riesgo por lo que se recomienda estudio.") 

def main():
    file_path = "/Users/ajelandro/Documents/GitHub/mLconnumpy/fetal_health.csv"
    data, X, y = get_fetal_dataset(file_path, 0.3)
    get_distribution_plots(data, list(X.columns))
    print('About to train RL...')
    y_test, y_pred, accuracy, weights, scaler = train_rl(X, y, 0.01, 10000)
    predict(X, weights, scaler)

if __name__ == '__main__':
    main()