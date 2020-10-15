import tensorflow as tf
from modules.feature_extractor import VGG16, ResNet50, EfficientNetB0
from modules.sequence_modeling import BidirectionalLSTM
from modules.predictor import Decoder

class Model:
    def __init__(self, input_shape=(64, 416, 3), max_sequence_length=12, charset_length=11, feature_extractor='Efficient', sequence_modeling='Bilstm'):
        # ** Feature Extractor ** #
        if feature_extractor == 'Efficient':
            self.feature_extractor = EfficientNetB0(input_shape=input_shape)
        elif feature_extractor == 'VGG':
            self.feature_extractor = VGG16(input_shape)
        elif feature_extractor == 'ResNet':
            self.feature_extractor = ResNet50(input_shape=input_shape)
        else: 
            print("Unknown feature extractor !")
            exit(-1)

        feature_map_channel = self.feature_extractor.channel #1280
        # ** Siquence Modeling ** #
        if sequence_modeling is None:
            self.sequence_model = None
        else:
            self.sequence_model = BidirectionalLSTM(input_size=feature_map_channel)
        
        # ** Prediction ** #
        self.predictor = Decoder(input_size=1280, lstm_units=128, max_sequence_length=12, charset_length=11)

    def make_model(self):
        inp = self.feature_extractor.model.input #(Ih, Iw, 3)
        feature_map = self.feature_extractor.model.output #(wxh, channel)
        print('Feature map shape: ',feature_map.shape) # Efficient: Nonex26x1280
        sequence_vectors = feature_map if self.sequence_model is None else self.sequence_model(feature_map)  
        print('Sequence output shape: ', sequence_vectors.shape) #Nonex26x1280
        output = self.predictor(sequence_vectors)
        print(output.shape) 
        model = tf.keras.models.Model(inputs=inp, outputs=output)
        return model 

if __name__ == '__main__':
    model = Model(input_shape=(64, 416, 3), max_sequence_length=12, charset_length=11,
                feature_extractor='Efficient', sequence_modeling='Bilstm').make_model()
    model.summary()

    