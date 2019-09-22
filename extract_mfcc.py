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


DEFAULT_INPUT_DIR = 'raw_wav'
DEFAULT_OUTPUT_DIR = 'raw_csv'
WINFUNC = lambda x: np.hamming(x)

def get_non_silence_idx_range_from_pitch(wav_name:str, pitch_csv_path:str) -> [int, int]:
    corresponding_pitch_csv = f'pitch-{wav_name.replace(".wav", ".csv")}'
    df = pd.read_csv(f'{pitch_csv_path}/{corresponding_pitch_csv}')

    candidate = []
    start_idx = 0
    end_idx = df.shape[0] + 1
    count = 0
    is_counting = False
    i = 0
    for i in range(len(df['F0'])):
        is_counting = df['F0'][i] != 0
        if is_counting:
            if count == 0:
                start_idx = i
            count += 1
            end_idx = i
        elif not is_counting and count > 0:
            candidate.append((start_idx, end_idx))
            count = 0
            start_idx = i
            end_idx = i
    candidate.append((start_idx, end_idx))

    (final_start, final_end) = max(candidate, key=lambda el: el[1] - el[0])
    (time_start, time_end) = (df.iloc[final_start]['Time'], df.iloc[final_end]['Time'])
    return time_start, time_end

def convert_pitch_idx_to_mfcc_idx(idx:int):
    OFFSET = 2 
    return idx + OFFSET

def main() -> None:
    try:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    except IndexError:
        print('Usage:')
        print('\targ1 = input_dir')
        print('\targ2 = output_dir')
        print('No specific directories specified. Using default values')
        input_dir = DEFAULT_INPUT_DIR
        output_dir = DEFAULT_OUTPUT_DIR

    PITCH_DIR_NAME = 'pitch'
    MFCC_DIR_NAME = 'mfcc'
    FEATURE_TYPE_COUNT = 3

    try:
        os.makedirs(f'{output_dir}/{MFCC_DIR_NAME}')
    except FileExistsError:
        pass

    for path, dirs, files in os.walk(input_dir):
        for file in files:
            if('.wav' in file):
                print(f'Processing {file}...')
                pitch_csv_path = f'{output_dir}/{PITCH_DIR_NAME}'
                time_start, time_end = get_non_silence_idx_range_from_pitch(file, pitch_csv_path)

                try:
                    (rate,sig) = wav.read(f'{path}/{file}')
                    frame_start = int(time_start * rate)
                    frame_end = int(time_end * rate)
                    if (len(sig.shape) >= 2):
                        if (sig.shape[1] == 2):
                            sig = np.mean(sig, axis=1)
                except ValueError as e:
                    print(e)
                    continue
                sig = sig[frame_start:frame_end]
                mfcc_feat = psf.mfcc(sig, rate, numcep=13, nfft=4096, winfunc=WINFUNC)
                d_mfcc_feat = psf.delta(mfcc_feat, 2)
                d_d_mfcc_feat = psf.delta(d_mfcc_feat, 2)

                final_mfcc_feat = np.concatenate(
                    (mfcc_feat.mean(axis=0), d_mfcc_feat.mean(axis=0), d_d_mfcc_feat.mean(axis=0)), 
                    axis=0
                    )

                name_mfcc = f'{output_dir}/{MFCC_DIR_NAME}/{MFCC_DIR_NAME}-{file.replace(".wav","").upper()}.csv'
                with open(name_mfcc, 'w', newline='') as mfccfile:
                    csv_out=csv.writer(mfccfile)
                    column_label = []
                    for i in range(FEATURE_TYPE_COUNT):
                        for j in range(final_mfcc_feat.shape[0] // FEATURE_TYPE_COUNT):
                            if i == 0:
                                if j == 0:
                                    label_name = f'energy_coef'
                                else:
                                    label_name = f'ceps_coef{j+1}'
                            elif i == 1:
                                if j == 0:
                                    label_name = f'delta_energy_coef'
                                else:
                                    label_name = f'delta_ceps_coef{j+1}'
                            elif i == 2:
                                if j == 0:
                                    label_name = f'delta_delta_energy_coef'
                                else:
                                    label_name = f'delta_delta_ceps_coef{j+1}'
                            column_label.append(label_name)
                    csv_out.writerow(column_label)
                    csv_out.writerow(final_mfcc_feat)

if __name__ == '__main__':
    main()