import uuid
from .data.io import DataFile, CorpusManager
import sklearn
from sklearn.model_selection import train_test_split  
import tensorflow as tf
from keras.layers import Dense, LSTM, Input, Masking, concatenate, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda, Dropout, Add, BatchNormalization, Flatten
from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model


class Gregarious(object):
    def __init__(self, df, df_seed=None, seed_model=None, optimizer="SGD", loss="binary_crossentropy", metrics=['mae', 'acc']):
        self.dataFile = df
        self.manager = CorpusManager(df)
        if df_seed:
            manager_seed = CorpusManager(df_seed).compute()
            self.__computed_data = self.manager.compute(conform_lengths=manager_seed["meta"]["lengths"])
            self.__seed = manager_seed["meta"]["lengths"]
        else:
            self.__computed_data = self.manager.compute()
            self.__seed = None
        if seed_model:
            self.model = load_model(seed_model)
            if not df_seed:
                manager_seed = CorpusManager(df).compute()
                self.__computed_data = self.manager.compute(conform_lengths=manager_seed["meta"]["lengths"])
                self.__seed = manager_seed["meta"]["lengths"]
        else:
            self.model = self.__build(self.__computed_data["meta"]["lengths"])
            self.model.compile(optimizer, loss, metrics)

    def recompile(self, optimizer="SGD", loss="binary_crossentropy", metrics=['mae', 'acc']):
        self.model.compile(optimizer, loss, metrics)

    def predict(self, handles, names, descs, statuses, f=False):
        predict_obj = self.manager.generate_predict_object({"handle":handles, "name":names, "description":descs, "status":statuses}, conform_lengths=self.__seed, f=f)
        return self.model.predict(x=predict_obj)

    def train(self, epochs=10, batch_size=10, validation_split=0, callbacks=None, save=None):
        X_train, X_test, y_train, y_test = [[], [], [], []], [[], [], [], []], [], []
        X_train[0], X_test[0], y_train, y_test = train_test_split(self.__computed_data["ins"][0], self.__computed_data["out"][0], test_size=validation_split, random_state=42)
        X_train[1], X_test[1], _, _ = train_test_split(self.__computed_data["ins"][1], self.__computed_data["out"][0], test_size=validation_split)
        X_train[2], X_test[2], _, _ = train_test_split(self.__computed_data["ins"][2], self.__computed_data["out"][0], test_size=validation_split, random_state=42)
        X_train[3], X_test[3], _, _ = train_test_split(self.__computed_data["ins"][3], self.__computed_data["out"][0], test_size=validation_split, random_state=42)
        # self.model.fit(x=self.__computed_data["ins"], y=self.__computed_data["out"], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, y_test), callbacks=callbacks)
        self.model.fit(x=X_train, y=[y_train], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, [y_test]), callbacks=callbacks)

        if save:
            self.model.save(save)

    def __build(self, lengths):
        handles_len = lengths[0]
        names_len = lengths[1]
        desc_len = lengths[2]
        status_len = lengths[3]
#         desc_len = lengths[0]
        # status_len = lengths[1]
        # fnf_len = lengths[4]
        # fnf = Input(shape=(2,))

        # handles_masked = Masking()(handles) 
        # names_masked = Masking()(names) 
        # desc_masked = Masking()(desc) 
        # status_masked = Masking()(status) 
        
        # handles_rec = LSTM(36)(handles_masked)
        # names_rec = LSTM(36)(names_masked)
        # desc_rec = LSTM(36)(desc_masked)
        # status_rec = LSTM(36)(status_masked)
        # net = self.__conv_unit(status_conv)
        # net = self.__conv_unit(net)
        # net = self.__conv_unit(net)
        # net = MaxPooling1D()(net)

        # net = Dropout(0.5)(net)

#         # net = self.__conv_unit(net)
        # net = self.__conv_unit(net)
        # net = self.__conv_unit(net)
        # net = MaxPooling1D()(net)

        # # net = Dropout(0.2)(net)

        # net = self.__conv_unit(net)
        # net = Dropout(0.2)(net)

        # net = self.__conv_unit(net)

        handles = Input(shape=(handles_len,))
        handles_normalized = BatchNormalization()(handles)
        names = Input(shape=(names_len,))
        names_normalized = BatchNormalization()(names)
        desc = Input(shape=(desc_len,))
        desc_normalized = BatchNormalization()(desc)
        status = Input(shape=(status_len,))
        status_normalized = BatchNormalization()(status)

        handles_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(handles_normalized)

        handles_conv_bigram = Conv1D(36, 10, padding="same", kernel_regularizer=regularizers.l2(1e-15))(handles_expanded)
        handles_conv_trigram = Conv1D(36, 15, padding="same", kernel_regularizer=regularizers.l2(1e-15))(handles_expanded)
        handles_conv_quadgram = Conv1D(36, 20, padding="same", kernel_regularizer=regularizers.l2(1e-15))(handles_expanded)

        handles_conv = MaxPooling1D()(concatenate([handles_conv_bigram, handles_conv_trigram, handles_conv_quadgram])) 

        names_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(names_normalized)

        names_conv_bigram = Conv1D(36, 10, padding="same", kernel_regularizer=regularizers.l2(1e-15))(names_expanded)
        names_conv_trigram = Conv1D(36, 15, padding="same", kernel_regularizer=regularizers.l2(1e-15))(names_expanded)
        names_conv_quadgram = Conv1D(36, 20, padding="same", kernel_regularizer=regularizers.l2(1e-15))(names_expanded)

        names_conv = MaxPooling1D()(concatenate([names_conv_bigram, names_conv_trigram, names_conv_quadgram])) 
        desc_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(desc_normalized)

        desc_conv_bigram = Conv1D(36, 10, padding="same")(desc_expanded)
        desc_conv_trigram = Conv1D(36, 15, padding="same")(desc_expanded)
        desc_conv_quadgram = Conv1D(36, 20, padding="same")(desc_expanded)

        desc_conv = MaxPooling1D()(concatenate([desc_conv_bigram, desc_conv_trigram, desc_conv_quadgram]))

        status_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(status_normalized)

        status_conv_bigram = Conv1D(36, 10, padding="same")(status_expanded)
        status_conv_trigram = Conv1D(36, 15, padding="same")(status_expanded)
        status_conv_quadgram = Conv1D(36, 20, padding="same")(status_expanded)

        status_conv = MaxPooling1D()(concatenate([status_conv_bigram, status_conv_trigram, status_conv_quadgram]))

        network_cat = concatenate([handles_conv, names_conv, desc_conv, status_conv], axis=1)
        # network_cat = concatenate([desc_conv, status_conv], axis=1)
        network_normalized = BatchNormalization()(network_cat)
        # network_cat = concatenate([desc_conv, status_conv], axis=1)

        # net = self.__conv_unit(network_normalized)
        # net = self.__conv_unit(net)
        # net = self.__conv_unit(net)
        # net = MaxPooling1D()(net)
        # net = self.__conv_unit(net)
        # net = self.__conv_unit(net)
        # net = self.__conv_unit(net)
        # net = MaxPooling1D()(net)

        # net = LSTM(25, return_sequences=True)(net)
        # mp = LSTM(25)(net)
        mp = Flatten()(network_normalized)

        # Dense, Group 1 
        net_ga1 = Dense(64, activation="relu")(mp)
        net_ga1 = Dense(32, activation="relu")(net_ga1)

        net_gb1 = Dense(64, activation="relu")(mp)
        net_gb1 = Dense(32, activation="relu")(net_gb1)

        net_g1 = Add()([net_ga1, net_gb1])

        # Dense, Group 2 
        net_ga2 = Dense(64, activation="relu")(mp)
        net_ga2 = Dense(32, activation="relu")(net_ga2)

        net_gb2 = Dense(64, activation="relu")(mp)
        net_gb2 = Dense(32, activation="relu")(net_gb2)

        net_g2 = Add()([net_ga2, net_gb2])

        # Dense, Direct
        net_gc = Dense(32, activation="relu")(mp)

        net = concatenate([net_g1, net_g2, net_gc])
        net = Dense(8, activation="relu")(net)

        net = Dense(2, name="result", activation="sigmoid", kernel_regularizer=regularizers.l2(1.5e-5))(net)

        # net = Dropout(1e-2)(net)
        
        return Model(inputs=[handles, names, desc, status], outputs=net)
        # return Model(inputs=[desc, status], outputs=net)

    def __conv_unit(self, in_layer):
        net = Conv1D(5, 5, padding="same")(in_layer)
        net = Conv1D(5, 10, padding="same")(net)
        net = Conv1D(5, 20, padding="same")(net)
        return net



