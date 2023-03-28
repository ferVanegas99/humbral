import numpy as np

# Función para entrenar el perceptrón
def perceptron_train(x, y, lr, epochs):
    # Inicializar pesos sinápticos y umbral
    w = np.zeros(x.shape[1])
    b = 0
    
    for epoch in range(epochs):
        # Iterar sobre todos los ejemplos de entrenamiento
        for i in range(x.shape[0]):
            # Calcular el producto punto entre los pesos y las entradas
            z = np.dot(w, x[i]) + b
            
            # Aplicar la función de activación
            if z > 0:
                y_hat = 1
            else:
                y_hat = 0
                
            # Actualizar los pesos y el umbral si la predicción es incorrecta
            if y_hat != y[i]:
                w += lr * (y[i] - y_hat) * x[i]
                b += lr * (y[i] - y_hat)
    
    return w, b

# Generar datos de entrenamiento aleatorios para la compuerta OR
x_or = np.random.rand(100, 2)
y_or = np.logical_or(x_or[:, 0] > 0.5, x_or[:, 1] > 0.5).astype(int)

# Entrenar el perceptrón para la compuerta OR
w_or, b_or = perceptron_train(x_or, y_or, lr=0.1, epochs=10)

# Generar datos de entrenamiento aleatorios para la compuerta NOT
x_not = np.random.rand(100, 1)
y_not = np.logical_not(x_not[:, 0] > 0.5).astype(int)

# Entrenar el perceptrón para la compuerta NOT
w_not, b_not = perceptron_train(x_not, y_not, lr=0.1, epochs=10)

# Generar datos de entrenamiento aleatorios para la compuerta AND
x_and = np.random.rand(100, 2)
y_and = np.logical_and(x_and[:, 0] > 0.5, x_and[:, 1] > 0.5).astype(int)

# Entrenar el perceptrón para la compuerta AND
w_and, b_and = perceptron_train(x_and, y_and, lr=0.1, epochs=10)

# Imprimir los pesos sinápticos y el umbral para cada compuerta
print("Pesos sinápticos y umbral para la compuerta OR:")
print("W:", w_or)
print("b:", b_or)

print("Pesos sinápticos y umbral para la compuerta NOT:")
print("W:", w_not)
print("b:", b_not)

print("Pesos sinápticos y umbral para la compuerta AND:")
print("W:", w_and)
print("b:", b_and)
