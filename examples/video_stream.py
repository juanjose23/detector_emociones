import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from emotion_processor.main import EmotionRecognitionSystem
from camera import Camera
import time

class VideoStream:
    def __init__(self, cam_index, emotion_recognition_system):
        self.camera = Camera(cam_index, 1280, 720)
        self.emotion_recognition_system = emotion_recognition_system

    def run(self):
        while True:
            ret, frame = self.camera.read()
            if ret:
                # Procesar el marco actual
                frame = self.emotion_recognition_system.frame_processing(frame)

                # Mostrar el marco en la ventana
                cv2.imshow('Emotion Recognition', frame)

                # Esperar a que se presione una tecla
                t = cv2.waitKey(5)
                if t == 27:  # Tecla ESC para salir
                    break
                elif t == ord('c'):  # Tecla 'c' para capturar la imagen
                    # Guardar la imagen
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f'captured_image_{timestamp}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Imagen capturada y guardada como '{filename}'.")
            else:
                print("No hay cámara conectada.")
        
        self.camera.release()
        cv2.destroyAllWindows()

def list_cameras():
    available_cameras = []
    for index in range(10):  # Intenta detectar hasta 10 cámaras
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    emotion_recognition_system = EmotionRecognitionSystem()
    
    # Listar cámaras disponibles
    cameras = list_cameras()
    
    if not cameras:
        print("No se encontraron cámaras.")
    else:
        print("Cámaras disponibles:")
        for i, cam_index in enumerate(cameras):
            print(f"{i + 1}: Cámara {cam_index}")
        
        # Solicitar al usuario que seleccione una cámara
        choice = int(input("Selecciona el número de la cámara que deseas usar: ")) - 1
        selected_camera_index = cameras[choice]

        video_stream = VideoStream(selected_camera_index, emotion_recognition_system)
        video_stream.run()