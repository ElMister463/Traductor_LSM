from dependencies import tf, load_model

# Ruta completa al archivo action.h5
model_path = r"C:\Users\ricar\Desktop\Traductor_LSM\action.h5"  # Usa 'r' para indicar que es una raw string
model = load_model(model_path)

print("Modelo cargado correctamente")

# Configurar el convertidor de TensorFlow Lite con opciones avanzadas
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Agregar soporte para operaciones selectas de TensorFlow si hay operaciones no compatibles con TFLite
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Operaciones estándar de TFLite
    tf.lite.OpsSet.SELECT_TF_OPS     # Operaciones selectas de TensorFlow
]

# Habilitar soporte para variables de recurso
converter.experimental_enable_resource_variables = True

# Deshabilitar la conversión experimental de listas de tensores
converter._experimental_lower_tensor_list_ops = False

# Convertir el modelo a TensorFlow Lite
try:
    tflite_model = converter.convert()
    print("Modelo convertido a TFLite correctamente")
    
    # Guardar el modelo TFLite
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Modelo TFLite guardado como 'model.tflite'")
except Exception as e:
    print(f"Error durante la conversión a TFLite: {e}")
