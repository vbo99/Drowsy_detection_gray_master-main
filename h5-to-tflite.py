import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file( 'trained_model/SqueezeNet_NIR.h5' )
converter.post_training_quantize = True
tflite_buffer = converter.convert()
open( 'trained_model/SqueezeNet_NIR_Quant.tflite' , 'wb' ).write( tflite_buffer )