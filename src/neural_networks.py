import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Input, Dense, Masking, Dropout, SpatialDropout1D, Layer
from keras.layers import LSTM, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras import constraints

#this layer takes takes an input with a shape of n and outputs the average as 1 dimensional vector
class GlobalAverage(Layer):
    def __init__(self, **kwargs):
        
        super(GlobalAverage, self).__init__(**kwargs)    

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
    
    def call(self, inputs, mask = None):
        denom = K.ones([inputs.shape[1]])

        return K.sum(inputs, axis=1,  keepdims=True) / K.sum(denom, keepdims=True)

    #need to implement so can save and reload
    def get_config(self):
        config = {}
        base_config = super(GlobalAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_model(n_hidden, embedding_matrix, max_len=300, dropout_rate=0.15, rnn_type="LSTM", train_embed=False):
    
    #if the embedding matrix is trainable, better up the regularizatoin
    if train_embed:
        dropout_rate *= 2

    embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=train_embed, name = 'embedding')
    sequence_input = Input(shape=(max_len,), dtype='int32', name='joke_seq')
    embedded_input = embedding_layer(sequence_input)
    mask1=Masking(mask_value=0., name='mask_paddings')(embedded_input)
    #drop out on words....
    #should switcth to new keras argument with shape...
    drop1 = SpatialDropout1D(dropout_rate, name='drop_words')(mask1)
    #more maskning
    mask2 = Masking(mask_value=0., name='mask_dropped_words')(drop1)
    
    if rnn_type=="LSTM":
        rnn = LSTM(n_hidden, implementation=2, unroll=True, name='reccurrent_layer', activation="tanh",
                  recurrent_dropout=dropout_rate*2, dropout=dropout_rate*2)(mask2)
    elif rnn_type=="GRU":
        rnn = GRU(n_hidden, implementation=2, unroll=True, name='reccurrent_layer', activation="tanh",
                  recurrent_dropout=dropout_rate, dropout=dropout_rate*2)(mask2)

    drop2 = Dropout(dropout_rate*2, name="drop_dense")(rnn)
    dense = Dense(int(n_hidden/2), activation="sigmoid", name="dense_sigmoid")(drop2)
    preds = GlobalAverage(name="avg_pred")(dense)

    model = Model(inputs=sequence_input, outputs=preds)
    
    return model


#train and valid should be tuples, with two elemnts
#first element is sequences, second is labels
def run_model(model, train, valid, out_path, patience=25, optimizer=Adam()):
    checkpointer = ModelCheckpoint(filepath=out_path,
                               monitor='val_loss',
                               verbose=1,
                               mode='min',
                               save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss',
                                min_delta=0.00001,
                                patience=patience,
                                verbose=1,
                                mode='min')
    model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])
    history = model.fit(x=train[0], y=train[1], epochs=1000, batch_size=2000,
                validation_data=(valid[0], valid[1]), callbacks=[checkpointer, earlystopper], verbose=1)
    return history