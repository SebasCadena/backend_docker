import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

RUTA_BASE = os.path.dirname(__file__)
CARPETA_IMAGENES = os.path.join(RUTA_BASE, "imagenes")
CARPETA_RESULTADOS = os.path.join(RUTA_BASE, "resultados")
CARPETA_LABELS = os.path.join(RUTA_BASE, "labels")


def crear_mascara_desde_txt(txt_path,   img_shape):
    """Crea máscara binaria a partir de anotaciones YOLO v8 (formato de segmentación)"""
    height, width = img_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3:  # Si no hay suficientes coordenadas
                continue

            try:
                # El primer valor es la clase, el resto son coordenadas
                coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                # Asegurar que tenemos pares de coordenadas
                if len(coords) % 2 != 0:
                    print(f"Advertencia: Número impar de coordenadas en {os.path.basename(txt_path)}")
                    coords = coords[:len(coords) // 2 * 2]  # Ajustar a número par

                # Convertir coordenadas normalizadas a píxeles
                polygon = coords.reshape(-1, 2) * [width, height]
                polygon = polygon.astype(np.int32)

                # Dibujar polígono relleno
                cv2.fillPoly(mask, [polygon], color=255)
            except Exception as e:
                print(f"Error procesando línea en {txt_path}: {str(e)}")
                continue

    return mask


def analizar_roya(img_path, txt_path):
    try:
        # 1. Cargar imagen
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Error al cargar imagen")

        # 2. Crear máscara desde el archivo YOLO
        mask_hoja = crear_mascara_desde_txt(txt_path, img.shape)
        if np.sum(mask_hoja) == 0:
            raise ValueError("Máscara vacía - Verificar archivo .txt")

        # 3. Aislar la hoja
        hoja_aislada = cv2.bitwise_and(img, img, mask=mask_hoja)

        # 4. Convertir a HSV para detección de color
        hsv = cv2.cvtColor(hoja_aislada, cv2.COLOR_BGR2HSV)

        # 5. Rangos para tonalidades de roya (ajustables)
        lower_roya = np.array([15, 50, 50])  # Amarillo claro
        upper_roya = np.array([30, 255, 255])  # Amarillo oscuro/café

        # 6. Detectar roya SOLO dentro de la hoja
        mask_roya = cv2.inRange(hsv, lower_roya, upper_roya)
        mask_roya = cv2.bitwise_and(mask_roya, mask_hoja)

        # 7. Cálculo de áreas
        area_total = cv2.countNonZero(mask_hoja)
        area_roya = cv2.countNonZero(mask_roya)
        porcentaje = (area_roya / area_total) * 100 if area_total > 0 else 0.0

        #7.5 Guardar txt
 
        os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
        nombre_base = os.path.splitext(os.path.basename(img_path))[0]
        txt_path_resultado = os.path.join(CARPETA_RESULTADOS,"por", f"{nombre_base}.txt")
        with open(txt_path_resultado, "w") as txt_file:
            txt_file.write(f"{porcentaje:.2f}%\n")
        print(f"📂 Archivo TXT guardado en: {txt_path_resultado}")

        # 8. Visualización
        plt.figure(figsize=(18, 6))

        # Original con contorno
        img_contorno = img.copy()
        contours, _ = cv2.findContours(mask_hoja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contorno, contours, -1, (0, 255, 0), 2)

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_contorno, cv2.COLOR_BGR2RGB))
        plt.title("Original con contorno YOLO")
        plt.axis('off')

        # Hoja aislada
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(hoja_aislada, cv2.COLOR_BGR2RGB))
        plt.title("Hoja aislada (YOLO)")
        plt.axis('off')

        # Roya detectada
        img_roya = img.copy()
        img_roya[mask_roya == 255] = [0, 0, 255]

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(img_roya, cv2.COLOR_BGR2RGB))
        plt.title(f"Roya detectada: {porcentaje:.2f}%")
        plt.axis('off')

        plt.tight_layout()

        # Guardar resultados
        os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
        nombre_base = os.path.splitext(os.path.basename(img_path))[0]
        plt.savefig(os.path.join(CARPETA_RESULTADOS, f"{nombre_base}_analisis.jpg"), bbox_inches='tight', dpi=150)
        plt.close()

        return {
            'porcentaje_roya': porcentaje,
            'area_total': area_total,
            'area_roya': area_roya
        }

    except Exception as e:
        print(f"Error procesando {img_path}: {str(e)}")
        return None


def procesar_lote():
    print("=== PROCESAMIENTO INICIADO ===")
    resultados = {}

    for img_file in os.listdir(CARPETA_IMAGENES):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(CARPETA_IMAGENES, img_file)
            txt_path = os.path.join(CARPETA_LABELS, f"{base_name}.txt")

            if not os.path.exists(txt_path):
                print(f"⚠️ Archivo .txt no encontrado para {img_file}")
                continue

            resultado = analizar_roya(img_path, txt_path)

            if resultado:
                resultados[img_file] = resultado
                print(f"✅ {img_file}: {resultado['porcentaje_roya']:.2f}% roya")

    return resultados


if __name__ == "__main__":
    resultados = procesar_lote()

    print("\n=== RESUMEN ===")
    for archivo, datos in resultados.items():
        print(f"{archivo}:")
        print(f"  - Área hoja: {datos['area_total']} px")
        print(f"  - Área roya: {datos['area_roya']} px")
        print(f"  - Porcentaje: {datos['porcentaje_roya']:.2f}%\n")

    print(f"🎉 Resultados guardados en: {CARPETA_RESULTADOS}")