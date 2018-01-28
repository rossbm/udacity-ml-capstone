import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Input, Dense, Masking, Dropout, SpatialDropout1D, Layer
from keras.layers import LSTM, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

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
    #def get_config(self):
        #config = {}
        #base_config = super(GlobalAverage, self).get_config()
        #return dict(list(base_config.items()) + list(config.items()))

class AUC_Callback(Callback):
    #code based on jamartinh's comment from https://github.com/keras-team/keras/issues/3230
    def __init__(self, train_data, validation_data, train_auc=True, batch_size=1000):
        super(AUC_Callback, self).__init__()
        self.x = train_data[0]
        self.y = train_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.batch_size = batch_size
        self.train_auc = train_auc
        
    def on_epoch_end(self, epoch, logs=None):
        if self.train_auc:
            y_pred = self.model.predict(self.x, self.batch_size)
            auc = roc_auc_score(self.y, y_pred)      
            print('auc: {0:.4%}'.format(auc))
    
        y_pred_val = self.model.predict(self.x_val, self.batch_size)
        val_auc = roc_auc_score(self.y_val, y_pred_val)
        print(' val_auc: {0:.4%}'.format(val_auc))

        #write to log
        if logs is not None:
            if self.train_auc:
                logs["auc"] = auc
            logs["val_auc"] = val_auc

def create_model(n_hidden, embedding_matrix, max_len=300, base_drop=0.15, rnn_type="LSTM", train_embed=False):
    #base drop is for words
    recurrent_drop = base_drop * 2
    dense_drop = recurrent_drop

    input_drop = recurrent_drop

    if embedding_matrix.shape[1] > 50:
        input_drop *= (4/3)
    
    if train_embed:
        input_drop *= (3/2)

    embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=max_len, mask_zero=True,
                            trainable=train_embed, name='embedding')

    sequence_input = Input(shape=(max_len,), dtype='int32', name='joke_seq')
    embedded_input = embedding_layer(sequence_input)
    mask1=Masking(mask_value=0., name='mask_paddings')(embedded_input)
    #drop out on words....
    #should switcth to new keras argument with shape...
    drop1 = SpatialDropout1D(base_drop, name='drop_words')(mask1)
    #more maskning
    mask2 = Masking(mask_value=0., name='mask_dropped_words')(drop1)
    
    if rnn_type=="LSTM":
        rnn = LSTM(n_hidden, implementation=2, unroll=True, name='reccurrent_layer', activation="tanh",
                  recurrent_dropout=recurrent_drop, dropout=input_drop)(mask2)
    elif rnn_type=="GRU":
        rnn = GRU(n_hidden, implementation=2, unroll=True, name='reccurrent_layer', activation="tanh",
                  recurrent_dropout=recurrent_drop, dropout=input_drop)(mask2)

    drop2 = Dropout(dense_drop, name="drop_dense")(rnn)
    dense = Dense(int(n_hidden/2), activation="sigmoid", name="dense_sigmoid")(drop2)
    preds = GlobalAverage(name="avg_pred")(dense)

    model = Model(inputs=sequence_input, outputs=preds)    
    return model


#train and valid should be tuples, with two elemnts
#first element is sequences, second is labels
def run_model(model, train, valid, out_path,patience=25, optimizer=Adam(), loss="binary_crossentropy", monitor='val_auc', batch_size=2000, metrics=["acc"], train_auc=True):
    checkpointer = ModelCheckpoint(filepath=out_path,
                               monitor=monitor,
                               verbose=1,
                               save_best_only=True,
                               mode="max")
    earlystopper = EarlyStopping(monitor=monitor,
                                min_delta=0.00001,
                                patience=patience,
                                verbose=1,mode="max")
    auc_monitor = AUC_Callback(train, valid, train_auc=train_auc, batch_size=batch_size)

    model.compile(loss=loss, optimizer=optimizer,  metrics=metrics)
    history = model.fit(x=train[0], y=train[1], epochs=1000, batch_size=batch_size,
                validation_data=(valid[0], valid[1]), callbacks=[auc_monitor, checkpointer, earlystopper], verbose=1)
    return history