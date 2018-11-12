"""
Authors:
    Juan Manuel Vera-Diaz, Daniel Pizarro, Javier Macias-Guarasa
Description:
    This code have been used to obtain the results shown in the paper "Towards end-to-End Acoustic Location
    Using Deep Learning: From Audio Signals to Source Position Coordinates". In case of using the code at any research,
    it must be cited the paper using the bibtex source given.
BibTex:
    @Article{s18103418,
        AUTHOR = {Vera-Diaz, Juan  Manuel and Pizarro, Daniel and Macias-Guarasa, Javier},
        TITLE = {Towards End-to-End Acoustic Localization Using Deep Learning: From Audio Signals to Source Position Coordinates},
        JOURNAL = {Sensors},
        VOLUME = {18},
        YEAR = {2018},
        NUMBER = {10},
        ARTICLE-NUMBER = {3418},
        URL = {http://www.mdpi.com/1424-8220/18/10/3418},
        ISSN = {1424-8220},
        ABSTRACT = {This paper presents a novel approach for indoor acoustic source localization using microphone arrays,
        based on a Convolutional Neural Network (CNN). In the proposed solution, the CNN is designed to directly estimate
        the three-dimensional position of a single acoustic source using the raw audio signal as the input information
        and avoiding the use of hand-crafted audio features. Given the limited amount of available localization data, we
        propose, in this paper, a training strategy based on two steps. We first train our network using semi-synthetic
        data generated from close talk speech recordings. We simulate the time delays and distortion suffered in the
        signal that propagate from the source to the array of microphones. We then fine tune this network using a small
        amount of real data. Our experimental results, evaluated on a publicly available dataset recorded in a real room,
        show that this approach is able to produce networks that significantly improve existing localization methods based
        on SRP-PHAT strategies and also those presented in very recent proposals based on Convolutional Recurrent Neural
        Networks (CRNN). In addition, our experiments show that the performance of our CNN method does not show a relevant
        dependency on the speaker's gender, nor on the size of the signal window being used.},
        DOI = {10.3390/s18103418}
    }
License:
    Copyright (C) <2018>  <Juan Manuel Vera-Diaz, Daniel Pizarro, Javier Macias-Guarasa>

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If not, see
    <https://www.gnu.org/licenses/>.
"""

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding, regularizers
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import glob
import wave
import struct
import scipy.io

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ADENET():
    def __init__(self, ModelPath, ModelName, DataDirTrain, DataDirTest, RoomRect,  vertexX, vertexY, vertexZ,listMicros, micPositions, DataDirTestMat=None, aWin=320, NEW=True):
        self.DataDirTrain = DataDirTrain        # Used .wav audio path for training and validation
        self.DataDirTest = DataDirTest          # Used .wav audio path for testing
        self.DataDirTestMat = DataDirTestMat    # Used data with info relative to real sequences (in .mat format)

        # Room size
        self.RoomRect = RoomRect
        self.vertexX = vertexX
        self.vertexY = vertexY
        self.vertexZ = vertexZ

        self.RoomLengthX = max(self.RoomRect[:, 0]) - min(self.RoomRect[:, 0])
        self.RoomLengthY = max(self.RoomRect[:, 1]) - min(self.RoomRect[:, 1])
        self.RoomLengthZ = 2400
        self.Roomx0 = min(self.RoomRect[:, 0])
        self.Roomy0 = min(self.RoomRect[:, 1])
        self.Roomz0 = -730

        # Microphones Set Up
        self.listMicros = listMicros
        self.Nm = len(self.listMicros)
        self.micPositions = micPositions

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121, projection='3d')
        ax2 = fig1.add_subplot(122)
        ax1.scatter(self.vertexX, self.vertexY, self.vertexZ, c='black')
        ax1.scatter(self.micPositions[0, :], self.micPositions[1, :], self.micPositions[2, :], c='green')
        ax2.scatter(self.micPositions[0, :], self.micPositions[1, :], c='green')
        plt.xlim((self.Roomx0, self.Roomx0 + self.RoomLengthX))
        plt.ylim((self.Roomy0, self.Roomy0 + self.RoomLengthY))
        fig1.suptitle('Microphone Location')
        plt.grid()
        plt.show()

        # Signal processing parameters
        self.Desired_SNR_dB = 30
        self.aWin = aWin                        # Length of the used windows (ms)
        self.Fs = 16000
        self.winSize = (self.aWin*self.Fs)/1000

        # Number of synthetic positions to simulate in train, validation and test phases
        self.Npos_train = 720
        self.Nwin_train = 10

        self.Npos_val = 80
        self.Nwin_val = 10

        self.Npos_test = 10
        self.Nwin_test = 1


        self.batch_size = 100
        self.steps_epoch = (self.Npos_train * self.Nwin_train) / self.batch_size
        self.validation_setps = (self.Npos_val * self.Nwin_val) / self.batch_size

        self.model_load_path = ModelPath + ModelName + '_' + str(self.aWin) + 'ms.hdf5'
        self.model_val_path = ModelPath + ModelName + '_' + str(self.aWin) + 'ms.hdf5'
        # self.model_train_path = ModelPath + ModelName + '_' + str(self.aWin) + 'ms_train.hdf5'

        self.optimizer = Adam(lr=0.001, decay=1e-5)
        self.Nepoch = 200

        self.input_shape = (self.winSize, self.Nm)
        self.output_size = 3

        if NEW:
            print 'Training from scratch ADENet model'
            self.adenet = self.load_ADENET_model()
        else:
            print 'Loading ADENet model'
            self.adenet = load_model(self.model_load_path)

        if self.DataDirTestMat is not None:
            mat = scipy.io.loadmat(self.DataDirTestMat)
            static_gt = mat['static_gt']

            p3d_aux = static_gt['p3d']
            p3d = p3d_aux[0, 0]
            self.pxyz = 1000 * p3d[0:3, :]

            sp_seg_aux = static_gt['sp_seg']
            self.sp_seg = np.round(self.Fs * sp_seg_aux[0, 0])

            pos_ind_aux = static_gt['pos_ind']
            pos_ind = pos_ind_aux[0, 0]
            self.pos_ind = pos_ind[0]

            self.winPerPos = np.round((self.sp_seg[1, :] - self.sp_seg[0, :]) / self.winSize)

    def load_ADENET_model(self):
        """
        Builds the used neural network model

        :return: neural network model
        """
        model = Sequential()
        model.add(Conv1D(96, 7, activation='relu', input_shape=self.input_shape, kernel_initializer='random_uniform',
                         bias_initializer='zeros'))
        model.add(MaxPooling1D(7))
        model.add(Conv1D(96, 7, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Conv1D(128, 5, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(128, 5, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(128, 3, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Flatten())
        model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_size, activation='linear'))

        return model

    def delayseq(self, x, delay_samples):
        """
        Delays an audio frame

        :param x: Audio frame to dalay
        :param delay_samples: number of samples to delay. Usually is a float number
        :return: Delayed audio frame
        """
        delay_int = int(round(delay_samples))

        nfft = self.nextpow2(len(x) + delay_int)

        fbins = 2 * np.pi * np.fft.ifftshift((np.arange(nfft) - nfft // 2)) / nfft

        X = np.fft.fft(x, nfft)
        Xs = np.fft.ifft(X * np.exp(-1j * delay_samples * fbins))

        if np.isreal(x[0]):
            Xs = Xs.real

        xs = np.zeros_like(x)
        xs[delay_int:] = Xs[delay_int:len(x)]

        return xs

    def nextpow2(self, n):
        return 2 ** (int(n) - 1).bit_length()

    def SNR_set(self, signal, Desired_SNR_dB):
        """
        Add a white gaussian noise to a given signal with a fixed SNR

        :param signal: Signal to add the noise
        :param Desired_SNR_dB: SNR target
        :return: signal + noise
        """
        Npts = len(signal)
        Noise = np.random.randn(1, Npts)

        Signal_Power = np.sum(np.abs(signal) * np.abs(signal)) / Npts
        Noise_Power = np.sum(np.abs(Noise) * np.abs(Noise)) / Npts

        K = (Signal_Power / Noise_Power) * 10 ** (-Desired_SNR_dB / 10)

        New_Noise = np.sqrt(K) * Noise

        return signal + New_Noise

    def create_signals(self, Nwin, DataDir):
        """
        From a given "synthetic" audio signal, this function simulates the delays from a random position to each used
        microphone.

        :param Nwin: Number of frames to extract from the used "synthetic" audio signal
        :param DataDir: Path where the "synthetic" audio signals are stored
        :return: simulated signals at each micropone and the random position associated
        """

        # Constants definition
        vs = 343 * 1000  # Sound speed at room temperature
        files = glob.glob(DataDir)  # Load all wave file paths
        nFiles = len(files)  # Number of files in the used directory

        signal = np.zeros((Nwin, self.winSize, self.Nm))  # Normalized wave signal extracted form a .wav file
        pos = np.zeros((Nwin, self.output_size))  # Random location of the signal source
        n = np.arange(self.winSize)

        length = 0
        while length <= 3 * self.winSize:  # All files with a length less than the limit are discarded
            # Open a random audio file and extract the signal
            randomFileIndex = np.random.randint(0, nFiles - 1)
            randomFile = files[randomFileIndex]
            waveFile = wave.open(randomFile, 'r')  # Open as reading file
            length = waveFile.getnframes()  # Number of samples in the signal

        signal_all = []
        for i in range(0, length):
            waveData = waveFile.readframes(1)
            data = struct.unpack("<h", waveData)
            signal_all.append(int(data[0]))

        # Generates a random position for the signal
        posX = float(self.Roomx0 + self.RoomLengthX * np.random.rand(1))
        posY = float(self.Roomy0 + self.RoomLengthY * np.random.rand(1))
        posZ = float(190 + 610 * np.random.rand(1))

        for AudioWin in range(0, Nwin):

            pos[AudioWin, :] = np.array([posX, posY, posZ])

            # Random window start position
            winStart = np.random.randint(0, length - 3 * self.winSize)
            winEnd = winStart + 3 * self.winSize

            # Windowed signal
            windowedSignal = signal_all[winStart:winEnd]

            DistancesMics = np.sqrt(np.sum((self.micPositions - np.transpose([pos[AudioWin, :]])) ** 2, axis=0))

            SamplesAllShifts = (DistancesMics / vs) * self.Fs
            SamplesAllShifts = SamplesAllShifts - min(SamplesAllShifts)

            desfase = np.random.rand(1) * np.pi
            Fnoise = np.random.rand(1) * 10 + 20
            Amp = np.random.rand(1) * 0.02 + 0.01

            noiseSin = Amp * np.sin((2 * np.pi * n * Fnoise / self.Fs) + (desfase * Fnoise / self.Fs))

            for mic in range(0, self.Nm):
                AmpTotal = np.random.rand(1) * 0.75 + 0.25
                y_delayed = self.delayseq(windowedSignal, SamplesAllShifts[mic])
                signal[AudioWin, :, mic] = y_delayed[self.winSize:2 * self.winSize]

                signal_max = 2 ** 15  # wav encodes with signed 16 bit

                signal_aux = signal[AudioWin, :, mic] / signal_max
                signal_aux = signal_aux + noiseSin
                signal[AudioWin, :, mic] = AmpTotal * self.SNR_set(signal_aux, self.Desired_SNR_dB)

        pos[:, 0] = (pos[:, 0] - self.Roomx0) / self.RoomLengthX
        pos[:, 1] = (pos[:, 1] - self.Roomy0) / self.RoomLengthY
        pos[:, 2] = (pos[:, 2] - self.Roomz0) / self.RoomLengthZ

        return signal, pos

    def create_dataset_synth(self, Npos, Nwin, DataDir):
        """
        Create a set of mic signals from a synthetic one and its associated random positions

        :param Npos: Number of random positions to simulate
        :param Nwin: Number of frames por position
        :param DataDir: Path where the synthetic audios are stored
        :return: signals and positions
        """
        x_train = np.zeros((Npos * Nwin, self.winSize, self.Nm))
        y_train = np.zeros((Npos * Nwin, self.output_size))

        for pos in range(0, Npos * Nwin, Nwin):
            x_batch, pos_batch = self.create_signals(Nwin, DataDir)
            x_train[pos:pos + Nwin, :, :] = x_batch
            y_train[pos:pos + Nwin, :] = pos_batch

        return x_train, y_train

    def create_dataset_real(self):
        """
        Load a set of mic signals from a IDIAP room and its associated positions

        :return: signals and positions
        """
        Nwin = int(np.sum(self.winPerPos))

        micAudio = []

        signal = np.zeros((Nwin, self.winSize, self.Nm))  # Normalized wave signal extracted form a .wav file
        positions = np.zeros((Nwin, self.output_size))
        NwinPos = np.zeros((1, np.array(self.pxyz).shape[1]))
        IntSup = np.zeros((1, np.array(self.pxyz).shape[1]))
        IntInf = np.zeros((1, np.array(self.pxyz).shape[1]))
        IntMed1 = np.zeros((1, np.array(self.pxyz).shape[1]))
        IntMed2 = np.zeros((1, np.array(self.pxyz).shape[1]))

        for index in range(0, len(self.pos_ind)):
            NwinPos[0, (self.pos_ind[index] - 1)] += self.winPerPos[index]

        for index in range(0, np.array(self.pxyz).shape[1]):
            if index == 0:
                IntInf[0, index] = 0
                IntSup[0, index] = NwinPos[0, index] - 1
                IntMed1[0, index] = np.round(((IntSup[0, index] - IntInf[0, index] + 1) / 3)) - 1
                IntMed2[0, index] = np.round(2 * ((IntSup[0, index] - IntInf[0, index] + 1) / 3)) - 1
            else:
                IntInf[0, index] = IntSup[0, index - 1] + 1
                IntSup[0, index] = IntInf[0, index] + NwinPos[0, index] - 1
                IntMed1[0, index] = np.round(((IntSup[0, index] - IntInf[0, index] + 1) / 3)) - 1 + IntInf[0, index]
                IntMed2[0, index] = np.round(2 * ((IntSup[0, index] - IntInf[0, index] + 1) / 3)) - 1 + IntInf[0, index]

        for mic in range(0, self.Nm):
            if self.listMicros[mic] == 0:
                waveFile = wave.open(self.DataDirTest + '_array1_mic1.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 1:
                waveFile = wave.open(self.DataDirTest + '_array1_mic2.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 2:
                waveFile = wave.open(self.DataDirTest + '_array1_mic3.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 3:
                waveFile = wave.open(self.DataDirTest + '_array1_mic4.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 4:
                waveFile = wave.open(self.DataDirTest + '_array1_mic5.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 5:
                waveFile = wave.open(self.DataDirTest + '_array1_mic6.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 6:
                waveFile = wave.open(self.DataDirTest + '_array1_mic7.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 7:
                waveFile = wave.open(self.DataDirTest + '_array1_mic8.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 8:
                waveFile = wave.open(self.DataDirTest + '_array2_mic1.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 9:
                waveFile = wave.open(self.DataDirTest + '_array2_mic2.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 10:
                waveFile = wave.open(self.DataDirTest + '_array2_mic3.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 11:
                waveFile = wave.open(self.DataDirTest + '_array2_mic4.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 12:
                waveFile = wave.open(self.DataDirTest + '_array2_mic5.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 13:
                waveFile = wave.open(self.DataDirTest + '_array2_mic6.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 14:
                waveFile = wave.open(self.DataDirTest + '_array2_mic7.wav', 'r')  # Open as reading file
            elif self.listMicros[mic] == 15:
                waveFile = wave.open(self.DataDirTest + '_array2_mic8.wav', 'r')  # Open as reading file

            length = waveFile.getnframes()  # Number of samples in the signal

            signal_all = []
            for i in range(0, length):
                waveData = waveFile.readframes(1)
                data = struct.unpack("<h", waveData)
                signal_all.append(int(data[0]))

            micAudio.append(signal_all)

        micAudio = np.array(micAudio)

        pos = 0

        for index in range(0, self.sp_seg.shape[1]):
            for winIndex in range(0, int(self.winPerPos[index])):
                winStart = int(self.sp_seg[0, index] + self.winSize * winIndex)
                winEnd = int(winStart + self.winSize)

                PosX = (float(self.pxyz[0, (self.pos_ind[index] - 1)]) - self.Roomx0) / self.RoomLengthX
                PosY = (float(self.pxyz[1, (self.pos_ind[index] - 1)]) - self.Roomy0) / self.RoomLengthY
                PosZ = (float(self.pxyz[2, (self.pos_ind[index] - 1)]) - self.Roomz0) / self.RoomLengthZ

                positions[pos, :] = np.array([PosX, PosY, PosZ])

                for micIndex in range(0, self.Nm):
                    signal_aux = micAudio[micIndex, winStart:winEnd]
                    signal[pos, :, micIndex] = signal_aux
                    signal_aux = (signal[pos, :, micIndex] / 2 ** 15) * 3
                    signal[pos, :, micIndex] = signal_aux

                pos += 1

        return signal, positions

    def load_data(self, Npos, Nwin, DataDir):
        """
        Generator function for training and validation phase

        :param Npos: Number of position tu simulate
        :param Nwin: Number of frames per position
        :param DataDir: Path where the synthtic files are stored
        :return: batch
        """

        while True:
            (x, y) = self.create_dataset_synth(Npos, Nwin, DataDir)
            posPermutations = np.random.permutation(Npos * Nwin)

            for Nbatch in range(0, Npos * Nwin, self.batch_size):
                limInf = Nbatch
                limSup = Nbatch + self.batch_size
                x2_train = x[posPermutations[limInf:limSup], :, :]
                y2_train = y[posPermutations[limInf:limSup], :]
                yield (x2_train, y2_train)

    def plot_historical(self, historical):
        """
        Plot the loss and the val loss got in the training

        :param historical: Historical of the losses got in the training
        """

        # list all data in history
        print(historical.history.keys())

        # summarize history for loss
        plt.plot(historical.history['loss'])
        plt.plot(historical.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train(self):
        """
        Function used for training ADENet
        """
        self.adenet.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        self.adenet.summary()

        early = EarlyStopping(patience=int(np.round(0.1 * self.Nepoch)))
        model_save = ModelCheckpoint(self.model_val_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        used_callbacks = [model_save, early]

        historical = self.adenet.fit_generator(generator=self.load_data(self.Npos_train, self.Nwin_train, self.DataDirTrain), steps_per_epoch=self.steps_epoch, epochs=self.Nepoch, verbose=1, callbacks=used_callbacks, validation_data=self.load_data(self.Npos_val, self.Nwin_val, self.DataDirTrain), validation_steps=self.validation_setps, workers=1, shuffle=False)

        self.plot_historical(historical)

    def fine_tuning(self, synth=True):
        """
        Function used for fine tuning ADENet

        :param synth: If True, fine tuning will be executed using synthetic data, otherwise, it will be done with real
                      data
        :return:
        """
        self.adenet.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        self.adenet.summary()

        model_save = ModelCheckpoint(self.model_val_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        used_callbacks = [model_save]

        if synth:
            historical = self.adenet.fit_generator(generator=self.load_data(self.Npos_train, self.Nwin_train, self.DataDirTrain), steps_per_epoch=self.steps_epoch, epochs=self.Nepoch, verbose=1, callbacks=used_callbacks, validation_data=self.load_data(self.Npos_val, self.Nwin_val, self.DataDirTrain), validation_steps=self.validation_setps, workers=1, shuffle=False)
        else:
            (x, source_pos) = self.create_dataset_real()
            historical = self.adenet.fit(x, source_pos, batch_size=self.batch_size, epochs=self.Nepoch, verbose=1, validation_split=0.1, callbacks=used_callbacks)

        self.plot_historical(historical)

    def test_on_synth(self):
        """
        Function used for testing ADENet on synthetic data
        """
        print('Generating test data')
        (x_test, source_pos_test) = self.create_dataset_synth(self.Npos_test, self.Nwin_test, self.DataDirTest)

        score = self.adenet.evaluate(x_test, source_pos_test, batch_size=1)
        pos_pred = self.adenet.predict(x_test, batch_size=1, verbose=1)

        print('\n')
        print("%s : %f%%" % (self.adenet.metrics_names[0], score[0]))
        print("%s : %f%%" % (self.adenet.metrics_names[1], score[1]))
        print('\n')

        pos_pred[:, 0] = (pos_pred[:, 0] * self.RoomLengthX) + self.Roomx0
        pos_pred[:, 1] = (pos_pred[:, 1] * self.RoomLengthY) + self.Roomy0
        pos_pred[:, 2] = (pos_pred[:, 2] * self.RoomLengthZ) + self.Roomz0

        source_pos_test[:, 0] = (source_pos_test[:, 0] * self.RoomLengthX) + self.Roomx0
        source_pos_test[:, 1] = (source_pos_test[:, 1] * self.RoomLengthY) + self.Roomy0
        source_pos_test[:, 2] = (source_pos_test[:, 2] * self.RoomLengthZ) + self.Roomz0

        error_xyz_acum = 0
        error_xy_acum = 0

        Nsamples = source_pos_test.shape[0]

        for index in range(0, Nsamples):

            error_x = (source_pos_test[index, 0] - pos_pred[index, 0]) ** 2
            error_y = (source_pos_test[index, 1] - pos_pred[index, 1]) ** 2
            error_z = (source_pos_test[index, 2] - pos_pred[index, 2]) ** 2
            error_xyz = np.sqrt(error_x + error_y + error_z)
            error_xy = np.sqrt(error_x + error_y)
            error_xyz_acum += error_xyz
            error_xy_acum += error_xy

        print 'Average Location Error x-y-z (mm)'
        print error_xyz_acum / Nsamples

        print 'Average Location Error x-y (mm)'
        print error_xy_acum / Nsamples

        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(121, projection='3d')
        ax1.scatter(self.vertexX, self.vertexY, self.vertexZ, c='black')
        ax2 = fig1.add_subplot(122)

        ax1.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], c='blue')
        ax2.scatter(pos_pred[:, 0], pos_pred[:, 1], c='blue')
        ax1.scatter(source_pos_test[:, 0], source_pos_test[:, 1], source_pos_test[:, 2], c='red')
        ax2.scatter(source_pos_test[:, 0], source_pos_test[:, 1], c='red')

        ax1.scatter(self.micPositions[0, :], self.micPositions[1, :], self.micPositions[2, :], c='green')
        ax2.scatter(self.micPositions[0, :], self.micPositions[1, :], c='green')
        ax2.legend(['predicted positions', 'actual positions'])

        plt.xlim((self.Roomx0, self.Roomx0 + self.RoomLengthX))
        plt.ylim((self.Roomy0, self.Roomy0 + self.RoomLengthY))

        plt.grid()
        plt.show()

    def test_on_real(self):
        """
        Function used for testing ADENet on IDIAP data
        """
        print('Generating test data')
        (x_test, source_pos_test) = self.create_dataset_real()

        score = self.adenet.evaluate(x_test, source_pos_test, batch_size=1)
        pos_pred = self.adenet.predict(x_test, batch_size=1, verbose=1)

        print('\n')
        print("%s : %f%%" % (self.adenet.metrics_names[0], score[0]))
        print("%s : %f%%" % (self.adenet.metrics_names[1], score[1]))
        print('\n')

        pos_pred[:, 0] = (pos_pred[:, 0] * self.RoomLengthX) + self.Roomx0
        pos_pred[:, 1] = (pos_pred[:, 1] * self.RoomLengthY) + self.Roomy0
        pos_pred[:, 2] = (pos_pred[:, 2] * self.RoomLengthZ) + self.Roomz0

        source_pos_test[:, 0] = (source_pos_test[:, 0] * self.RoomLengthX) + self.Roomx0
        source_pos_test[:, 1] = (source_pos_test[:, 1] * self.RoomLengthY) + self.Roomy0
        source_pos_test[:, 2] = (source_pos_test[:, 2] * self.RoomLengthZ) + self.Roomz0

        error_xyz_acum = 0
        error_xy_acum = 0

        Nsamples = source_pos_test.shape[0]

        for index in range(0, Nsamples):
            error_x = (source_pos_test[index, 0] - pos_pred[index, 0]) ** 2
            error_y = (source_pos_test[index, 1] - pos_pred[index, 1]) ** 2
            error_z = (source_pos_test[index, 2] - pos_pred[index, 2]) ** 2
            error_xyz = np.sqrt(error_x + error_y + error_z)
            error_xy = np.sqrt(error_x + error_y)
            error_xyz_acum += error_xyz
            error_xy_acum += error_xy

        print 'Average Location Error x-y-z (mm)'
        print error_xyz_acum / Nsamples

        print 'Average Location Error x-y (mm)'
        print error_xy_acum / Nsamples

        self.pxyz = np.array(self.pxyz)

        for posIndex in range(0, self.pxyz.shape[1]):
            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(121, projection='3d')
            ax1.scatter(self.vertexX, self.vertexY, self.vertexZ, c='black')
            ax2 = fig1.add_subplot(122)

            for index in range(0, source_pos_test.shape[0]):
                if (int(self.pxyz[0, posIndex]) == int(source_pos_test[index, 0])) and (int(self.pxyz[1, posIndex]) == int(source_pos_test[index, 1])):
                    ax1.scatter(pos_pred[index, 0], pos_pred[index, 1], pos_pred[index, 2], c='blue')
                    ax2.scatter(pos_pred[index, 0], pos_pred[index, 1], c='blue')
                    ax1.scatter(source_pos_test[index, 0], source_pos_test[index, 1], source_pos_test[index, 2], c='red')
                    ax2.scatter(source_pos_test[index, 0], source_pos_test[index, 1], c='red')

            ax1.scatter(self.micPositions[:, 0], self.micPositions[:, 1], self.micPositions[:, 2], c='green')
            ax2.scatter(self.micPositions[:, 0], self.micPositions[:, 1], c='green')

            plt.xlim((self.Roomx0, self.Roomx0 + self.RoomLengthX))
            plt.ylim((self.Roomy0, self.Roomy0 + self.RoomLengthY))

            plt.title('Posicion ' + str(posIndex + 1))
            plt.grid()
            plt.pause(0.01)
            plt.clf()

if __name__ == '__main__':

    # Used .wav audio path
    DataDirTrain = 'cf/train/*wav'
    DataDirTestSynth = 'cf/test/*wav'
    DataDirTestReal = 'data/seq01-1p-0000/16kHz/seq01-1p-0000'
    DataDirMat = 'data/seq01-1p-0000/seq01-1p-0000_gt.mat'
    ModelPath = 'models/ADENET/'
    ModelName = 'ADENET_synth'

    # Room floor coordinates (mm)
    vertex0 = [1800, 2200, -730]
    vertex1 = [1800, -6000, -730]
    vertex2 = [-1800, -6000, -730]
    vertex3 = [-1800, 2200, -730]

    vertex00 = [1800, 2200, 1670]
    vertex11 = [1800, -6000, 1670]
    vertex22 = [-1800, -6000, 1670]
    vertex33 = [-1800, 2200, 1670]

    RoomRect = np.array([vertex0, vertex1, vertex2, vertex3])

    vertexX = [vertex0[0], vertex00[0], vertex1[0], vertex11[0], vertex2[0], vertex22[0], vertex3[0], vertex33[0]]
    vertexY = [vertex0[1], vertex00[1], vertex1[1], vertex11[1], vertex2[1], vertex22[1], vertex3[1], vertex33[1]]
    vertexZ = [vertex0[2], vertex00[2], vertex1[2], vertex11[2], vertex2[2], vertex22[2], vertex3[2], vertex33[2]]

    # Microphones Info
    listMicros = [0, 4, 10, 14]  # Used microphones ID @ setup 1

    mic0 = [-100, 400, 0]
    mic1 = [-71, 329, 0]
    mic2 = [0, 300, 0]
    mic3 = [71, 329, 0]
    mic4 = [100, 400, 0]
    mic5 = [71, 471, 0]
    mic6 = [0, 500, 0]
    mic7 = [-71, 471, 0]
    mic8 = [-100, -400, 0]
    mic9 = [-71, -471, 0]
    mic10 = [0, -500, 0]
    mic11 = [71, -471, 0]
    mic12 = [100, -400, 0]
    mic13 = [71, -329, 0]
    mic14 = [0, -300, 0]
    mic15 = [-71, -329, 0]
    mic16 = vertex0
    mic17 = vertex1
    mic18 = vertex2
    mic19 = vertex3

    micPositions = np.array([mic0, mic1, mic2, mic3, mic4, mic5, mic6, mic7, mic8, mic9, mic10, mic11, mic12, mic13, mic14, mic15, mic16, mic17, mic18, mic19])
    micUsedPositions = micPositions[listMicros, :]
    micPositions = np.transpose(micUsedPositions)

    aWin = 80	# Used audio window (in ms)

    Train = False
    FineTuning = True
    TestOnSynth = True
    TestOnReal = False

    if Train:
        ADENet = ADENET(ModelPath=ModelPath,
                        ModelName=ModelName,
                        DataDirTrain=DataDirTrain,
                        DataDirTest=None,
                        RoomRect=RoomRect,
                        vertexX=vertexX,
                        vertexY=vertexY,
                        vertexZ=vertexZ,
                        listMicros=listMicros,
                        micPositions=micPositions,
                        aWin=aWin,
                        NEW=True)

        ADENet.train()

    if FineTuning:
        ADENet = ADENET(ModelPath=ModelPath,
                        ModelName=ModelName,
                        DataDirTrain=DataDirTrain,
                        DataDirTest=None,
                        RoomRect=RoomRect,
                        vertexX=vertexX,
                        vertexY=vertexY,
                        vertexZ=vertexZ,
                        listMicros=listMicros,
                        micPositions=micPositions,
                        DataDirTestMat=DataDirMat,
                        aWin=aWin,
                        NEW=False)

        ADENet.fine_tuning(synth=True)

    if TestOnSynth:
        ADENet = ADENET(ModelPath=ModelPath,
                        ModelName=ModelName,
                        DataDirTrain=None,
                        DataDirTest=DataDirTestSynth,
                        RoomRect=RoomRect,
                        vertexX=vertexX,
                        vertexY=vertexY,
                        vertexZ=vertexZ,
                        listMicros=listMicros,
                        micPositions=micPositions,
                        aWin=aWin,
                        NEW=False)

        ADENet.test_on_synth()
    if TestOnReal:
        ADENet = ADENET(ModelPath=ModelPath,
                        ModelName=ModelName,
                        DataDirTrain=None,
                        DataDirTest=DataDirTestReal,
                        RoomRect=RoomRect,
                        vertexX=vertexX,
                        vertexY=vertexY,
                        vertexZ=vertexZ,
                        listMicros=listMicros,
                        micPositions=micPositions,
                        DataDirTestMat=DataDirMat,
                        aWin=aWin,
                        NEW=False)

        ADENet.test_on_real()
