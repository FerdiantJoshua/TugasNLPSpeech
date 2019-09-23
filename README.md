# Tugas 2 NLP Speech: Voice Recognition

## Script list
1. count_file.py  
Used for counting number of sound type (a:100, i:100, u:95... etc.)
2. extract_mfcc.py  
Used for extracting mfcc feature from wav files
3. extract_pitch_formant.py  
Used for extracting pitch and formant feature from wav files
4. move.py  
Used for moving files wav files according to its type (A91.wav -> raw_wav/a/A91.wav... etc.)
5. rename.py  
Used for renaming file using regex
6. Remove 12 Prefix Chars.bat  
Used for renaming file by omiting first 12 chars (Windows only)
7. Training_by_MFCC.ipynb  
Notebook for training

## extract_mfcc.py
### Usage
```sh
python extract_mfcc.py {path_to dir_containing_wav_files} {path_to_output_dir}
```
