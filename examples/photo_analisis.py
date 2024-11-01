import cv2
import os
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt

# Configuración de la ruta de la carpeta de imágenes y el archivo de salida
image_folder = "C:\\Users\\jrios\\Documents\\face-emotion-recognition\\examples"
output_csv = "emotion_analysis_results.csv"
reference_image_path = None  # Imagen de referencia para comparar identidad

# Lista para almacenar los resultados
results = []

# Configura la primera imagen como referencia para comparación de identidad
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        reference_image_path = os.path.join(image_folder, filename)
        break

# Iterar sobre cada archivo en la carpeta de imágenes
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        print(f"Analizando la imagen: {image_path}")
        
        try:
            # Analizar la imagen para detectar emociones
            result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                region = result[0]['region']
                
                # Comparar identidad con la imagen de referencia
                if reference_image_path:
                    is_same_person = DeepFace.verify(img1_path=reference_image_path, img2_path=image_path)['verified']
                else:
                    is_same_person = False  # Si no hay imagen de referencia

                # Almacenar los resultados en la lista
                results.append({
                    "Filename": filename,
                    "Dominant Emotion": emotion,
                    "Same Person": is_same_person
                })
                
                # Cargar y mostrar la imagen con los resultados
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Dibujar región facial y emoción
                if 'region' in result[0]:  # Verifica si hay una región
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image_rgb, f"{emotion} ({'Same' if is_same_person else 'Different'})", 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Mostrar la imagen en matplotlib
                plt.imshow(image_rgb)
                plt.axis('off')
                plt.title(f"{filename}: {emotion} - {'Same Person' if is_same_person else 'Different Person'}")
                plt.show()
                
            else:
                print(f"No se detectaron emociones en {filename}")
        
        except Exception as e:
            print(f"Error analizando {filename}: {e}")

# Guardar los resultados en un archivo CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Análisis completado y resultados guardados en '{output_csv}'")
