from __future__ import print_function, division
import random
import sys

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py
import tensorflow as tf

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout, Flatten
from keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam


from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.legacy.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class RNNDisaggregator(Disaggregator):


    def __init__(self, window_size=100):
        '''Initialize disaggregator
        '''
        self.MODEL_NAME = "WindowGRU"
        # self.mmax = None
        # # self.mmin = None
        # self.mmax_y = None
        # self.mmin_y = None
        self.sstd_x = None
        self.sstd_y = None
        self.mmean_x = None
        self.mmean_y = None
        self.MIN_CHUNK_LENGTH = window_size
        self.window_size = window_size
        self.model = self._create_model()

    def train(self, mains, meter, epochs=1, batch_size=256, **load_kwargs):
        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)
        
        history = {'loss': [], 'val_loss': []}

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        # if self.mmax == None:
        #     self.mmax = 6081.36
        # if self.mmax_y ==None:
        #     self.mmax_y= 2906.00

        '''Microwave 
        '''

        # if self.sstd_x == None:
        #     self.sstd_x = 364.28488
        # if self.sstd_y == None:
        #     self.sstd_y = 136.06915
        # if self.mmean_x == None:
        #     self.mmean_x = 221.91333
        # if self.mmean_y == None:
        #     self.mmean_y = 17.104923

        '''Fridge 
        '''

        if self.sstd_x == None:
            self.sstd_x = 356.2344
        if self.sstd_y == None:
            self.sstd_y = 84.37994
        if self.mmean_x == None:
            self.mmean_x = 219.84592
        if self.mmean_y == None:
            self.mmean_y = 61.480576        

        while(run):
            # mainchunk = self._normalize(mainchunk, self.mmax)
            # meterchunk = self._normalize(meterchunk, self.mmax_y)
            mainchunk = self._normalize(mainchunk, self.sstd_x,self.mmean_x)
            meterchunk = self._normalize(meterchunk, self.sstd_y,self.mmean_y)

            history_chunk = self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            history['loss'].extend(history_chunk.history['loss'])
            history['val_loss'].extend(history_chunk.history['val_loss'])
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False
        return history

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = np.array(mainchunk[ix])
        meterchunk = np.array(meterchunk[ix])

        indexer = np.arange(self.window_size)[None, :] + np.arange(len(mainchunk)-self.window_size+1)[:, None]
        mainchunk = mainchunk[indexer]
        meterchunk = meterchunk[self.window_size-1:]
        mainchunk = np.reshape(mainchunk, (mainchunk.shape[0], mainchunk.shape[1],1))

        print(mainchunk.shape)
        print(meterchunk.shape)
        print(self.sstd_x)
        print(self.sstd_y)
        print(self.mmean_x)
        print(self.mmean_y)
        print(batch_size)
        history = self.model.fit(mainchunk, meterchunk, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1)
        return history



    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):


        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = self._normalize(chunk, self.sstd_x,self.mmean_x)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power = self._denormalize(appliance_power,self.sstd_y,self.mmean_y)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):

        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        Y_len = len(X_batch)
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(X_batch)-self.window_size+1)[:, None]
        X_batch = X_batch[indexer]
        X_batch = np.reshape(X_batch, (X_batch.shape[0],X_batch.shape[1],1))

        pred = self.model.predict(X_batch, batch_size=256)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[self.window_size-1:Y_len], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers


    def import_model(self, filename):

        self.model = load_model(filename)
        with h5py.File(filename, 'a') as hf:
            ds = hf.get('disaggregator-data').get('sstd_x')
            self.sstd_x = np.array(ds)[0]
            self.sstd_y = np.array(hf.get('disaggregator-data').get('sstd_y'))[0]
            self.mmean_x = np.array(hf.get('disaggregator-data').get('mmean_x'))[0]
            self.mmean_y = np.array(hf.get('disaggregator-data').get('mmean_y'))[0]

    def export_model(self, filename):

        self.model.save(filename)
        with h5py.File(filename, 'a') as hf:
            gr = hf.create_group('disaggregator-data')
            gr.create_dataset('sstd_x', data = [self.sstd_x])
            gr.create_dataset('sstd_y', data=[self.sstd_y])
            gr.create_dataset('mmean_x', data=[self.mmean_x])
            gr.create_dataset('mmean_y', data=[self.mmean_y])

    def _normalize(self, chunk, stdd,meann):

        tchunk = (chunk-meann)/stdd
        return tchunk

    def _denormalize(self, chunk, stdd,meann):

        tchunk = chunk*stdd+meann
        return tchunk
    
#     def _normalize(self, chunk, maxx):

#         tchunk = chunk/maxx
#         return tchunk

#     def _denormalize(self, chunk, maxx):

#         tchunk = chunk*maxx
#         return tchunk

    def _create_model(self):
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(16, 4, activation="linear", input_shape=(self.window_size,1), padding="same", strides=1))

        #Bi-directional LSTMs
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))
        model.add(tf.keras.layers.Dropout(.2))
        # Fully Connected Layers
        model.add(tf.keras.layers.Dense(128, activation='tanh'))
        model.add(tf.keras.layers.Dense(1, activation='linear'))


        model.compile(loss='mse', optimizer='adadelta')
        model.summary()
                # adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)


        return model