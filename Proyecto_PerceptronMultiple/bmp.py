import struct, os, numpy as np

def bmp_to_array(path):
    with open(path, 'rb') as file:
        # Leer los primeros 54 bytes que contienen la cabecera del BMP
        header = file.read(54)
        
        # Extraer los datos de la cabecera
        file_type = struct.unpack('<2s', header[0:2])[0].decode('ascii')
        data_start = struct.unpack('<I', header[10:14])[0]  # Dirección de inicio de los datos de la imagen
        width = struct.unpack('<I', header[18:22])[0]  # Anchura de la imagen
        height = struct.unpack('<I', header[22:26])[0]  # Altura de la imagen
        pixel_size = struct.unpack('<H', header[28:30])[0]  # Tamaño de cada punto
        compression = struct.unpack('<I', header[30:34])[0]  # Compresión
        color_table_size = struct.unpack('<I', header[46:50])[0]  # Tamaño de la tabla de color
        
        # Comprobamos el tipo de archivo
        if file_type != 'BM':
            print("Este archivo no es un BMP válido.")
            return
        
        # Verificar compresión (0 = sin compresión)
        if compression != 0:
            print("Este script solo soporta BMP sin compresión.")
            return
        
        # Verificar que se maneja un formato de bits por píxel soportado (1, 4, 8, 24, 32)
        if pixel_size not in [1, 4, 8, 24, 32]:
            print(f"Formato de {pixel_size} bits por píxel no soportado.")
            return
        
        # Mover el puntero al inicio de los datos de la imagen
        file.seek(data_start)
        
        # Leer los datos de los píxeles
        image = []
        if pixel_size == 8:
            image_data = file.read(width * height)
            
            for i in range(height):
                row = []
                for j in range(width):
                    b = image_data[(i * width + j)]
                    row.append(b)  
                image.append(row)
        
        elif pixel_size == 24:
            # 24 bits por píxel (BGR)
            image_data = file.read(width * height * 3)  # 3 bytes por píxel (BGR)
            for i in range(height):
                row = []
                for j in range(width):
                    b = image_data[(i * width + j) * 3]
                    g = image_data[(i * width + j) * 3 + 1]
                    r = image_data[(i * width + j) * 3 + 2]
                    row.append((r, g, b))  # Almacenar como RGB
                image.append(row)
        
        elif pixel_size == 32:
            # 32 bits por píxel (BGRA)
            image_data = file.read(width * height * 4)  # 4 bytes por píxel (BGRA)
            for i in range(height):
                row = []
                for j in range(width):
                    b = image_data[(i * width + j) * 4]
                    g = image_data[(i * width + j) * 4 + 1]
                    r = image_data[(i * width + j) * 4 + 2]
                    a = image_data[(i * width + j) * 4 + 3]  # Canal alfa
                    row.append((r, g, b, a))  # Almacenar como RGBA
                image.append(row)
        
        image = image[::-1]

        return image

def image_flatten(image):
    new_image = []
    for row in image:
        new_image.extend(row)
    return new_image

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            # Extraer el dígito del nombre del archivo (asumimos que está antes del '_')
            label = int(filename.split('_')[0])
            # Leer la imagen BMP como un archivo binario
            img_path = os.path.join(folder, filename)
            image =bmp_to_array(img_path)
            image =image_flatten(image)
            if len(image) != 784:
                print("!!!")
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
            
    return images, labels

