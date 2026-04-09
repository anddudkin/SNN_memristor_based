import librosa
file = "/home/anddudkin/PycharmProjects/SNN_memristor_based/data/Sound/database/airplane/files/3cc20963-fadc-4510-8411-ccb653b7708b.mp3"
# file = "/home/anddudkin/PycharmProjects/SNN_memristor_based/data/Sound/database/motors/files/3ef315b9-ae51-44a0-94aa-ed6b22ce51d8.mp3"
# file = "/home/anddudkin/PycharmProjects/SNN_memristor_based/data/Sound/database/motors/files/e5e8859b-da66-4a43-95d3-7998ae61ff2a.mp3"
#file = "/home/anddudkin/PycharmProjects/SNN_memristor_based/data/Sound/database/airplane/files/ae7e18f0-8383-4363-b48a-d114d80dbe58.mp3"
signal, sr = librosa.load(file, sr = 22050)
print(signal)

import matplotlib.pyplot as plt
import librosa.display as ld
import numpy as np
plt.figure(figsize=(12,4))
ld.waveshow(signal, sr=sr)
plt.show()  #амплитуда/время


n_fft = 2048
ft = np.abs(librosa.stft(signal[:n_fft], hop_length = n_fft+1))
plt.plot(ft)
plt.title('Spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')
plt.show() #амплитуда/частота


X = librosa.stft(signal)
s = librosa.amplitude_to_db(abs(X))
ld.specshow(s, sr=sr, x_axis = 'time', y_axis='linear')
plt.colorbar()
plt.show() #спектрограмма частота/время

mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc = 40, hop_length=512)
melspectrum = librosa.feature.melspectrogram(y=signal, sr = sr,
                                        	hop_length =512, n_mels = 40)
plt.show()