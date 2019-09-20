import os 
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import python_speech_features as psf
import seaborn as sns
import scipy.io.wavfile as wav


INPUT_DIR = 'raw_wav'
OUTPUT_DIR = 'raw_csv'
PITCH_DIR_NAME = 'pitch'
FORMANT_DIR_NAME = 'formant'

try:
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
except IndexError:
	print('Usage:')
	print('\targ1 = input_dir')
	print('\targ2 = output_dir')
	print('No specific directories specified. Using default value')
	input_dir = INPUT_DIR
	output_dir = OUTPUT_DIR

try:
	os.makedirs(f'{OUTPUT_DIR}/{FORMANT_DIR_NAME}')
	os.makedirs(f'{OUTPUT_DIR}/{PITCH_DIR_NAME}')
except FileExistsError:
	pass

for folder in os.listdir(input_dir):
	for voice in os.listdir(f'{input_dir}/{folder}'):
		print(voice)
		if('.wav' in voice):
			snd = parselmouth.Sound(f'{input_dir}/{folder}/{voice}')
			pitch = snd.to_pitch()
			formant = snd.to_formant_burg()
			pitch_values = pitch.selected_array['frequency']
			
			pitch_tuple = []
			formant_tuple = []
			for i,v in enumerate(pitch_values):
				time = pitch.get_time_from_frame_number(int(i+1))
				pitch_tuple.append((time,pitch_values[i]))
			numPoints = formant.get_number_of_frames()
			for point in range(0, numPoints):
				point += 1
				time = formant.get_time_from_frame_number(point)
				f1 = formant.get_value_at_time(1, time)
				f2 = formant.get_value_at_time(2, time)
				f3 = formant.get_value_at_time(3, time)
				f4 = formant.get_value_at_time(4, time)
				f5 = formant.get_value_at_time(5, time)
				b1 = formant.get_bandwidth_at_time(1, time)
				b2 = formant.get_bandwidth_at_time(2, time)
				b3 = formant.get_bandwidth_at_time(3, time)
				b4 = formant.get_bandwidth_at_time(4, time)
				b5 = formant.get_bandwidth_at_time(5, time)
				nformants = 0
				if(str(f1) != "nan"):
					nformants += 1
				if(str(f2) != "nan"):
					nformants += 1
				if(str(f3) != "nan"):
					nformants += 1
				if(str(f4) != "nan"):
					nformants += 1
				if(str(f5) != "nan"):
					nformants += 1
				formant_tuple.append((time, nformants, f1, b1, f2, b2, f3, b3, f4, b4, f5, b5))
			name_pitch = "pitch/" + "pitch-" + voice.replace('.wav','').upper() + ".csv"
			with open(name_pitch, 'w', newline='') as pitchfile:
				csv_out=csv.writer(pitchfile)
				csv_out.writerow(['Time','F0'])
				for row in pitch_tuple:
					csv_out.writerow(row)
			name_formant = "formant/" + "formant-" + voice.replace('.wav','').upper() + ".csv"
			with open(name_formant, 'w', newline='') as formantfile:
				csv_out=csv.writer(formantfile)
				csv_out.writerow(['time(s)','nformants','F1(Hz)','B1(Hz)','F2(Hz)','B2(Hz)','F3(Hz)','B3(Hz)','F4(Hz)','B4(Hz)','F5(Hz)','B5(Hz)'])
				for row in formant_tuple:
					csv_out.writerow(row)
