import bmp, perceptron as p, numpy as np
from sklearn.preprocessing import StandardScaler

folder_path = 'dataset'

# Cargar las imágenes y las etiquetas
X, y = bmp.load_images_from_folder(folder_path)

# Normalizar las imágenes
scaler = StandardScaler()
X = scaler.fit_transform(X)

perceptron = p.MLP(input_size=784, output_size=10)
perceptron.train(X, y, epochs=1000)

testImage = bmp.bmp_to_array('prueba.bmp')
testImage = bmp.image_flatten(testImage)
# Asegurarte de que sea un array NumPy con forma (1, 784)
testImage = np.array(testImage).reshape(1, -1)  
testImage = scaler.transform(testImage)

print(f"El perceptron predice: {perceptron.predict(testImage)}")
