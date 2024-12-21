import numpy as np

# Función para convertir las etiquetas a formato one-hot
def to_one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

# Definición de la clase del perceptrón
class MLP:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Inicialización de pesos
        self.bias = np.zeros((1, output_size))  # Sesgo

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def sigmoid(self, z):
        # Limitar los valores de entrada para evitar desbordamientos
        z = np.clip(z, -10, 10)  # Limita los valores 
        return 1 / (1 + np.exp(-z))  # Función de activación sigmoide


    def train(self, X, y, epochs=1000, learning_rate=0.01):
        # Convertir las etiquetas en formato one-hot
        y_one_hot = to_one_hot(y, self.output_size)

        # Entrenamiento
        for epoch in range(epochs):
            # Propagación hacia adelante
            output = self.forward(X)

            # Cálculo de la pérdida (error cuadrático medio)
            loss = np.mean(np.square(y_one_hot - output))  # Error cuadrático medio

            # Retropropagación (gradientes)
            output_error = output - y_one_hot
            d_weights = np.dot(X.T, output_error) / len(X)
            d_bias = np.mean(output_error, axis=0, keepdims=True)

            # Actualización de los pesos y sesgo
            self.weights -= learning_rate * d_weights
            self.bias -= learning_rate * d_bias

            # Mostrar el progreso
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

    def predict(self, X):
        # Realizar la predicción
        output = self.forward(X)
        
        # Devolver la clase con la probabilidad más alta
        return np.argmax(output, axis=1)  # El índice con la mayor probabilidad es la clase predicha
