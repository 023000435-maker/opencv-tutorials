import cv2
import numpy as np
import os  # Asegura que las rutas funcionen sin importar desde dónde ejecutes

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Read the first frame and convert to float
_, img = cap.read()
averageValue1 = np.float32(img)

while True:
    # Capture next frame
    _, img = cap.read()
    
    # Update background model
    cv2.accumulateWeighted(img, averageValue1, 0.02)
    
    # Convert back to 8-bit for display
    resultingFrames1 = cv2.convertScaleAbs(averageValue1)

    # Show both original and background model
    cv2.imshow('Original Frame', img)
    cv2.imshow('Background (Running Average)', resultingFrames1)
    
    # Exit on Esc key
    if cv2.waitKey(30) & 0xFF == 27:
        # Obtener el directorio base (opencv-tutorials)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Definir rutas de guardado
        ruta_original = os.path.join(base_dir, 'data', 'original.jpg')
        ruta_procesada = os.path.join(base_dir, 'out', 'outbackground.jpg')
        
        # 2. Guardar ambas imágenes
        cv2.imwrite(ruta_original, img)             # Guarda la foto original en 'data'
        cv2.imwrite(ruta_procesada, resultingFrames1) # Guarda el fondo procesado en 'out'
        
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()