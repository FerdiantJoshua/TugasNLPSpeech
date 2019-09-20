import os 
import csv
import sys
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


try:
	main_path = sys.argv[1]
	wav_name = sys.argv[2]
except IndexError:
	print('Usage:')
	print('\targ1 = main_path')
	print('\targ2 = wav_name')
	exit

pitch_dir_name = 'pitch'
formant_dir_name = 'formant'
mfcc_dir_name = 'mfcc'

try:
    os.mkdir(mfcc_dir_name)
except FileExistsError:
    pass

for path, dirs, files in os.walk(main_path):
	for file in files:
		if(file == wav_name):
			(rate,sig) = wav.read(f'{path}/{file}')
			mfcc_feat = mfcc(sig,rate)
			d_mfcc_feat = delta(mfcc_feat, 2)
			d_d_mfcc_feat = delta(d_mfcc_feat, 2)
			fbank_feat = logfbank(sig,rate)

			print(f'rate, sig: {rate},{sig}')
			print(f'mfcc_feat shape: {mfcc_feat.shape}')
			print(f'd_mfcc_feat shape: {d_mfcc_feat.shape}')
			print(f'd_d_mfcc_feat shape: {d_d_mfcc_feat.shape}')
			print(f'fbank_feat shape: {fbank_feat.shape}')
			print(mfcc_feat[1:3,:])
			print()
			print(d_mfcc_feat[1:3,:])
			print()
			print(d_d_mfcc_feat[1:3,:])
			print()
			print(fbank_feat[1:3,:])