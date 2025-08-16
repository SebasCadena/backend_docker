from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
import logging
import torch
from typing import Optional
from fastapi import Form
import os
from roya import procesar_lote
from typing import List
import base64
import glob

app = FastAPI()

# Configuración avanzada de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173/", "https://trabajo-de-grado-c13a2.web.app", "https://d0a0-190-107-19-227.ngrok-free.app", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_credentials=True,
    max_age=3600
)

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carga del modelo con verificación
try:
    logger.info("⏳ Cargando modelo YOLO...")
    modelo = YOLO("full30.pt")
    torch.device('cpu')
    logger.info("✅ Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"❌ Error cargando el modelo: {str(e)}")
    raise RuntimeError("No se pudo cargar el modelo YOLO") from e


# ------------------------------------------------------------------

    
@app.get("/") #PRUEBA EL ESTADO DEL API
def health_check():
    return JSONResponse(
        content={
            "status": "API activa",
            "modelo": "cargado" if modelo else "no cargado"
        },
        status_code=200
    )


# ------------------------------------------------------------------


@app.get("/run-roya-analysis")  #EJECUTA ROYA.py
def run_roya_analysis():
    try:
        # Llama a la función principal del script
        resultados = procesar_lote()

        # Devuelve los resultados en formato JSON
        return JSONResponse(
            content={
                "status": "success",
                "resultados": resultados
            },
            status_code=200
        )
    except Exception as e:
        # Manejo de errores
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e)
            },
            status_code=500
        )
    
# ------------------------------------------------------------------


@app.post("/segment-leaf") # Devuelve 1 sola segmentaciòn
async def segment_leaf(file: UploadFile = File(...),filename: str = Form(...)):
    try:
        logger.info(f"📥 Recibiendo archivo: {filename}")
        
        # Verificación básica del archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Solo se permiten imágenes")

        contents = await file.read()
        logger.info(f"📏 Tamaño de imagen recibida: {len(contents) / 1024:.2f} KB")

        # Decodificación de la imagen
        nparr = np.frombuffer(contents, np.uint8)
        imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if imagen is None:
            logger.error("⚠️ No se pudo decodificar la imagen")
            raise HTTPException(400, "Formato de imagen no soportado")

        logger.info(f"🖼️ Dimensión de imagen: {imagen.shape}")

        # Procesamiento con YOLO
        logger.info("🔍 Ejecutando predicción...")
        resultados = modelo.predict(
            imagen,
            imgsz=640,
            conf=0.4,
            device="cpu"
        )
        logger.info("🎯 Predicción completada")

        # Procesamiento de máscaras
        if not hasattr(resultados[0], "masks") or resultados[0].masks is None:
            logger.warning("⚠️ No se detectaron máscaras")
            raise HTTPException(400, "No se detectaron objetos en la imagen")

        mascaras = resultados[0].masks.data.cpu().numpy()
        logger.info(f"🔄 Procesando {len(mascaras)} máscaras...")

        # --- Generar archivo TXT con las coordenadas ---
        txt_data = []
        for cls, mask in zip(resultados[0].boxes.cls.cpu().numpy(), resultados[0].masks.xyn):
            cls = int(cls)
            puntos = ["{:.6f}".format(coord) for punto in mask for coord in punto]
            linea = f"{cls} " + " ".join(puntos)
            txt_data.append(linea)

        # Crear carpeta labels si no existe
        labels_dir = "labels"
        os.makedirs(labels_dir, exist_ok=True)
        
        # Guardar el archivo TXT en el sistema de archivos
        txt_filename = os.path.join(labels_dir, f"{filename.rsplit('.', 1)[0]}.txt")
        with open(txt_filename, "w") as txt_file:
            txt_file.write("\n".join(txt_data))
        logger.info(f"📂 Archivo TXT guardado en: {txt_filename}")

        # --- Continuar con el procesamiento visual original ---
        mascara_mas_grande = max(
            ((m * 255).astype("uint8") for m in mascaras),
            key=lambda m: cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].size,
            default=None
        )

        if mascara_mas_grande is None:
            logger.warning("⚠️ Máscara más grande no encontrada")
            raise HTTPException(400, "No se pudo segmentar la imagen")

        mascara_rsz = cv2.resize(mascara_mas_grande, (imagen.shape[1], imagen.shape[0]))
        overlay = np.zeros_like(imagen)
        overlay[np.where(mascara_rsz > 0)] = (0, 255, 0)  # Verde
        imagen_segmentada = cv2.addWeighted(imagen, 0.7, overlay, 0.3, 0)

        # Crear carpeta imagenes si no existe
        imagenes_dir = "imagenes"
        os.makedirs(imagenes_dir, exist_ok=True)

        # Guardar la imagen segmentada en el sistema de archivos
        segmented_image_filename = os.path.join(imagenes_dir, f"{filename.rsplit('.', 1)[0]}.png")
        cv2.imwrite(segmented_image_filename, imagen)
        logger.info(f"📂 Imagen segmentada guardada en: {segmented_image_filename}")

        # Codificar la imagen segmentada para devolverla
        _, img_encoded = cv2.imencode(".png", imagen_segmentada)

        # Devolver la imagen segmentada
        return StreamingResponse(
            io.BytesIO(img_encoded),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=segmentada_{file.filename}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔥 Error crítico: {str(e)}", exc_info=True)
        raise HTTPException(500, "Error interno procesando la imagen") from e
    

    # ------------------------------------------------------------------


@app.post("/process-multiple-images")
async def process_multiple_images(files: List[UploadFile] = File(...)):
    try:
        resultados = []
        logger.info(f"📥 Recibiendo {len(files)} archivos para procesar")

        for file in files:
            logger.info(f"Procesando archivo: {file.filename}")
            try:
                # Verificación del tipo de archivo
                if not file.content_type.startswith('image/'):
                    logger.warning(f"Archivo no válido: {file.filename}")
                    resultados.append({
                        "filename": file.filename,
                        "error": "Archivo no válido (no es una imagen)",
                        "status": "error"
                    })
                    continue

                # Leer el contenido del archivo
                contents = await file.read()
                
                # Decodificar la imagen
                nparr = np.frombuffer(contents, np.uint8)
                imagen = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if imagen is None:
                    logger.warning(f"No se pudo decodificar la imagen: {file.filename}")
                    resultados.append({
                        "filename": file.filename,
                        "error": "No se pudo decodificar la imagen",
                        "status": "error"
                    })
                    continue


                original_image_path = os.path.join("imagenes", file.filename)
                cv2.imwrite(original_image_path, imagen)
                logger.info(f"📂 Imagen original guardada en: {original_image_path}")
                



                # Procesamiento con YOLO
                logger.info(f"🔍 Ejecutando predicción para {file.filename}...")
                resultados_yolo = modelo.predict(
                    imagen,
                    imgsz=640,
                    conf=0.4,
                    device="cpu"
                )

                # Verificar si se detectaron máscaras
                if not hasattr(resultados_yolo[0], "masks") or resultados_yolo[0].masks is None:
                    logger.warning(f"No se detectaron máscaras en {file.filename}")
                    resultados.append({
                        "filename": file.filename,
                        "error": "No se detectaron máscaras",
                        "status": "error"
                    })
                    continue

                mascaras = resultados_yolo[0].masks.data.cpu().numpy()
                



                txt_data = []
                for cls, mask in zip(resultados_yolo[0].boxes.cls.cpu().numpy(), resultados_yolo[0].masks.xyn):
                    cls = int(cls)
                    puntos = ["{:.6f}".format(coord) for punto in mask for coord in punto]
                    linea = f"{cls} " + " ".join(puntos)
                    txt_data.append(linea)

                txt_filename = os.path.join("labels", f"{file.filename.rsplit('.', 1)[0]}.txt")
                with open(txt_filename, "w") as txt_file:
                    txt_file.write("\n".join(txt_data))
                logger.info(f"📂 Archivo TXT guardado en: {txt_filename}")




                # Obtener la máscara más grande
                mascara_mas_grande = max(
                    ((m * 255).astype("uint8") for m in mascaras),
                    key=lambda m: cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].size,
                    default=None
                )

                if mascara_mas_grande is None:
                    resultados.append({
                        "filename": file.filename,
                        "error": "No se encontró una máscara válida",
                        "status": "error"
                    })
                    continue

                # Redimensionar la máscara y crear la superposición
                mascara_rsz = cv2.resize(mascara_mas_grande, (imagen.shape[1], imagen.shape[0]))
                overlay = np.zeros_like(imagen)
                overlay[np.where(mascara_rsz > 0)] = (0, 255, 0)  # Verde
                imagen_segmentada = cv2.addWeighted(imagen, 0.7, overlay, 0.3, 0)

                # Convertir a base64
                _, img_encoded = cv2.imencode(".png", imagen_segmentada)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                # Guardar en resultados
                resultados.append({
                    "filename": file.filename,
                    "imagen_segmentada_base64": img_base64,
                    "status": "success"
                })

                logger.info(f"✅ Procesamiento exitoso para {file.filename}")

            except Exception as e:
                logger.error(f"Error procesando {file.filename}: {str(e)}")
                resultados.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "error"
                })

        return JSONResponse(
            content={"status": "success", "resultados": resultados},
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error general en process_multiple_images: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


# ------------------------------------------------------------------


@app.delete("/clear-all-files")
def clear_all_files():
    """
    Endpoint para eliminar todos los archivos de las carpetas:
    - imagenes/
    - labels/
    - resultados/
    - resultados/por/
    """
    try:
        carpetas_a_limpiar = [
            "imagenes",
            "labels", 
            "resultados",
            os.path.join("resultados", "por")
        ]
        
        archivos_eliminados = []
        errores = []
        
        for carpeta in carpetas_a_limpiar:
            try:
                # Verificar si la carpeta existe
                if not os.path.exists(carpeta):
                    logger.warning(f"📁 La carpeta {carpeta} no existe")
                    continue
                
                # Obtener todos los archivos en la carpeta
                patron = os.path.join(carpeta, "*")
                archivos = glob.glob(patron)
                
                # Filtrar solo archivos (no carpetas)
                archivos = [f for f in archivos if os.path.isfile(f)]
                
                logger.info(f"🗑️ Eliminando {len(archivos)} archivos de {carpeta}")
                
                for archivo in archivos:
                    try:
                        os.remove(archivo)
                        archivos_eliminados.append(archivo)
                        logger.info(f"✅ Eliminado: {archivo}")
                    except Exception as e:
                        error_msg = f"Error eliminando {archivo}: {str(e)}"
                        errores.append(error_msg)
                        logger.error(f"❌ {error_msg}")
                        
            except Exception as e:
                error_msg = f"Error procesando carpeta {carpeta}: {str(e)}"
                errores.append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        # Preparar respuesta
        respuesta = {
            "status": "success",
            "mensaje": f"Proceso completado. {len(archivos_eliminados)} archivos eliminados.",
            "archivos_eliminados": len(archivos_eliminados),
            "detalles": {
                "archivos_eliminados": archivos_eliminados,
                "errores": errores
            }
        }
        
        if errores:
            respuesta["status"] = "partial_success"
            respuesta["mensaje"] += f" {len(errores)} errores encontrados."
            
        logger.info(f"🧹 Limpieza completada: {len(archivos_eliminados)} archivos eliminados, {len(errores)} errores")
        
        return JSONResponse(
            content=respuesta,
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"🔥 Error crítico en clear_all_files: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "status": "error",
                "mensaje": "Error interno al limpiar archivos",
                "error": str(e)
            },
            status_code=500
        )
