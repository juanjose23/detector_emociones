import cv2
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt

# Configuración del archivo de salida
output_csv = "emotion_analysis_results.csv"

# Lista para almacenar los resultados
results = []

# Inicializa la captura de video
cap = cv2.VideoCapture(0)  # Cambia a la URL de la cámara si usas una cámara IP

# Comprueba si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

try:
    while True:
        # Captura frame a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir el frame. Saliendo...")
            break

        # Procesar la imagen
        print("Analizando el frame...")
        try:
            # Analizar el cuadro para detectar emociones
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if isinstance(result, list) and len(result) > 0:
                emotion = result[0]['dominant_emotion']
                region = result[0]['region']
                
                # Almacenar los resultados en la lista
                results.append({
                    "Dominant Emotion": emotion
                })

                # Dibujar región facial y emoción
                if 'region' in result[0]:  # Verifica si hay una región
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Mostrar el cuadro en tiempo real
                cv2.imshow('Emotion Recognition', frame)

            else:
                print("No se detectaron emociones en el frame actual.")

        except Exception as e:
            print(f"Error analizando el frame: {e}")

        # Romper el bucle con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Análisis interrumpido por el usuario.")

# Guardar los resultados en un archivo CSV al finalizar
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Análisis completado y resultados guardados en '{output_csv}'")

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
