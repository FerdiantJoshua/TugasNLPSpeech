import parselmouth
import os 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys

try:
	main_path = sys.argv[1]
	wav_name = sys.argv[2]
except IndexError:
	print('Usage:')
	print('\targ1 = main_path')
	print('\targ2 = wav_name')
	exit

save_dir = 'raw_csv'
pitch_dir_name = 'pitch'
formant_dir_name = 'formant'
mfcc_dir_name = 'mfcc'

try:
    os.makedirs('{save_dir}/{mfcc_dir_name}')
except FileExistsError:
    pass

for path, dirs, files in os.walk(main_path):
	for file in files:
		if(file == wav_name):
			snd = parselmouth.Sound(f'{path}/{file}')
			print(snd)
			mfcc = snd.to_mfcc()
			print(mfcc)
			
			mfcc_features = mfcc.extract_features()
			print(mfcc_features)
			time = np.array([i for i in range(0, mfcc_features.get_number_of_columns())])
			mfcc_features = np.vstack((time, mfcc_features.as_array())).tolist()
			name_mfcc = f'{save_dir}/{mfcc_dir_name}/{mfcc_dir_name}-{file.replace(".wav","").upper()}.csv'
			print(name_mfcc)
			with open(name_mfcc, 'w', newline='') as mfccfile:
				csv_out=csv.writer(mfccfile)
				for row in mfcc_features:
					csv_out.writerow(row)

			# mfcc_features = np.vstack((time, mfcc_features.as_array())).transpose()

			# name_mfcc = f'{mfcc_dir_name}/{mfcc_dir_name}-{file.replace(".wav","").upper()}.csv'
			# with open(name_mfcc, 'w', newline='') as mfccfile:
			# 	csv_out=csv.writer(mfccfile)
			# 	csv_out.writerow(['frame', 'feature1', 'feature2', 'feature3', 'feature4'])
			# 	for row in mfcc_features:
			# 		csv_out.writerow(row)

# try:
#     os.mkdir(formant_dir_name)
#     os.mkdir(pitch_dir_name)
# except FileExistsError:
#     pass

# for folder in os.listdir(main_path):
# 	for voice in os.listdir(f'{main_path}/{folder}'):
# 		print(voice)
# 		if('.wav' in voice):
# 			snd = parselmouth.Sound(f'{main_path}/{folder}/{voice}')
# 			pitch = snd.to_pitch()
# 			formant = snd.to_formant_burg()
# 			pitch_values = pitch.selected_array['frequency']
			
# 			pitch_tuple = []
# 			formant_tuple = []
# 			for i,v in enumerate(pitch_values):
# 				time = pitch.get_time_from_frame_number(int(i+1))
# 				pitch_tuple.append((time,pitch_values[i]))
# 			numPoints = formant.get_number_of_frames()
# 			for point in range(0, numPoints):
# 				point += 1
# 				time = formant.get_time_from_frame_number(point)
# 				f1 = formant.get_value_at_time(1, time)
# 				f2 = formant.get_value_at_time(2, time)
# 				f3 = formant.get_value_at_time(3, time)
# 				f4 = formant.get_value_at_time(4, time)
# 				f5 = formant.get_value_at_time(5, time)
# 				b1 = formant.get_bandwidth_at_time(1, time)
# 				b2 = formant.get_bandwidth_at_time(2, time)
# 				b3 = formant.get_bandwidth_at_time(3, time)
# 				b4 = formant.get_bandwidth_at_time(4, time)
# 				b5 = formant.get_bandwidth_at_time(5, time)
# 				nformants = 0
# 				if(str(f1) != "nan"):
# 					nformants += 1
# 				if(str(f2) != "nan"):
# 					nformants += 1
# 				if(str(f3) != "nan"):
# 					nformants += 1
# 				if(str(f4) != "nan"):
# 					nformants += 1
# 				if(str(f5) != "nan"):
# 					nformants += 1
# 				formant_tuple.append((time, nformants, f1, b1, f2, b2, f3, b3, f4, b4, f5, b5))
# 			name_pitch = "pitch/" + "pitch-" + voice.replace('.wav','').upper() + ".csv"
# 			with open(name_pitch, 'w', newline='') as pitchfile:
# 				csv_out=csv.writer(pitchfile)
# 				csv_out.writerow(['Time','F0'])
# 				for row in pitch_tuple:
# 					csv_out.writerow(row)
# 			name_formant = "formant/" + "formant-" + voice.replace('.wav','').upper() + ".csv"
# 			with open(name_formant, 'w', newline='') as formantfile:
# 				csv_out=csv.writer(formantfile)
# 				csv_out.writerow(['time(s)','nformants','F1(Hz)','B1(Hz)','F2(Hz)','B2(Hz)','F3(Hz)','B3(Hz)','F4(Hz)','B4(Hz)','F5(Hz)','B5(Hz)'])
# 				for row in formant_tuple:
# 					csv_out.writerow(row)
