import keras.models as KM
import tensorflow as tf
import coremltools
import mrcnn.coreml_model as modellib
from coremltools.proto import NeuralNetwork_pb2

from mrcnn.eti import EtiConfigT

config = EtiConfigT(0, 0)


def convert_lambda(layer):
    if layer.function == modellib.lambda_a:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'lambda_a'
    elif layer.function == modellib.lambda_b:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'lambda_b'
    elif layer.function == modellib.lambda_c:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'lambda_c'
    elif layer.function == modellib.lambda_d:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = 'lambda_d'
    else:
        params = None

    return params


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig(0, 0)
config.DETECTION_MIN_CONFIDENCE = 0.975
config.display()

model_path = 'mask_rcnn_eti_train_0044.h5'
model = modellib.MaskRCNN(mode="inference", model_dir='./',
                          config=config)

weights_path = model_path
print("Loading weights ", weights_path)
# load eti model weights
model.load_weights(weights_path, by_name=True)

model.keras_model.save('convert_1.h5')

output_labels = ['tutku']
new_model = KM.load_model('convert_1.h5', custom_objects={'ProposalLayer': modellib.ProposalLayer,
                                                          'PyramidROIAlign': modellib.PyramidROIAlign,
                                                          'DetectionLayer': modellib.DetectionLayer})
#print('input shape : ', new_model.input_shape)
#print('inputs : ', new_model.inputs)

#print('output shape : ', new_model.output_shape)
#print('outputs : ', new_model.outputs)

#new_model.summary()

convert_1 = coremltools.converters.keras.convert(new_model,
                                                 class_labels=output_labels,
                                                 add_custom_layers=True,
                                                 custom_conversion_functions={"Lambda": convert_lambda})

convert_1.author = 'Deniz Simsek'
convert_1.short_description = 'tutku recognition and segmentation'

convert_1.save('det_tutku_iphone.mlmodel')
