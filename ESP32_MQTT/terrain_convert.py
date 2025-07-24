#!/usr/bin/env python3
import tensorflow as tf

INPUT_H5      = 'terrain_cnn.h5'
OUTPUT_TFLITE = 'terrain_cnn.tflite'

def convert_model(input_h5, output_tflite):
    model = tf.keras.models.load_model(input_h5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_buf = converter.convert()
    with open(output_tflite, 'wb') as f:
        f.write(tflite_buf)
    print(f"Converted {input_h5} → {output_tflite}")

if __name__ == '__main__':
    convert_model(INPUT_H5, OUTPUT_TFLITE)
