'''
INPUT: RTTM
OUTPUT: WAV, TXT (TTS result)
'''

import os
import json
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from MeloTTS_w_noise.melo.api import TTS
from pydub import AudioSegment
import re 
import random
from huggingface_hub.utils._errors import HfHubHTTPError
import time
from tqdm import tqdm
import glob
import pkuseg
from janome.tokenizer import Tokenizer
import nltk
import argparse
from konlpy.tag import Okt

okt = Okt()

nltk.download('averaged_perceptron_tagger_eng') 

parser = argparse.ArgumentParser(
    description="Generate hdf5 files for cross-view AP evaluation from audio and forced alignment transcript."
)
parser.add_argument("--lang", type=str, default="kr", help="kr, en, jp, zh")
parser.add_argument("--mode", type=str, default="train", help="train, valid, test")
parser.add_argument("--device_num", type=int, default=0)
parser.add_argument("--morph", type=str, default="False", help="spliting per morph for Korean")
parser.add_argument("--out_dir", type=str, default="/home/yeonghwa/workspace/flitto/CrossviewAP_Share0522/datasets")
parser.add_argument("--openvoice_dir", type=str, default="/home/yeonghwa/workspace/flitto/OpenVoice_Share0522")
parser.add_argument("--refspk_path", type=str, default="/home/yeonghwa/workspace/flitto/CrossviewAP_Share0522/datasets/reference_audio/SJH.wav")
parser.add_argument("--rttm_dir", type=str, default="/home/yeonghwa/workspace/flitto/CrossviewAP_Share0522/datasets/rttm")
args = parser.parse_args()

lang = args.lang
mode = args.mode

# data maximum
if mode == 'train':
    end_length = 32 * 60 * 60 * 1000 # 32H
elif mode == 'valid':
    end_length = 3 * 60 * 60 * 1000 # 3H
else: 
    end_length = 32 * 60 * 60 * 1000 # 32H

# output dir
output_dir = os.path.join(args.out_dir, 'tts', mode, lang)
os.makedirs(output_dir, exist_ok=True)

morph_flag = args.morph.lower() in ("true", "1", "yes", "y")

if lang == 'kr' and morph_flag:
    output_dir = f'{output_dir}_morph'
    os.makedirs(output_dir, exist_ok=True)

# rttm path
if mode == 'test':
    if lang == 'zh':
        rttm_path = f"{args.rttm_dir}/{mode}/CN.rttm"
    elif lang == 'jp':
        rttm_path = f"{args.rttm_dir}/{mode}/JP.rttm"
    elif lang == 'en':
        rttm_path = f"{args.rttm_dir}/{mode}/EN.rttm"
    elif lang == 'kr':
        rttm_path = f"{args.rttm_dir}/{mode}/KR.rttm"
else:
    if lang == 'zh':
        rttm_path = f"/nas_homes/byeonggeuk/dataset/Flitto/IIPL_SD_CN_metadata/{mode}/rttm"
    elif lang == 'jp':
        rttm_path = f"/nas_homes/byeonggeuk/dataset/Flitto/IIPL_SD_JP_metadata/{mode}/rttm"
    elif lang == 'en':
        rttm_path = f"/nas_homes/byeonggeuk/dataset/Flitto/IIPL_SD_EN_metadata/{mode}/rttm"
    elif lang == 'kr':
        rttm_path = f"/nas_homes/byeonggeuk/dataset/Flitto/SD_2/{mode}/rttm"
# tokenizer
if lang == "zh":
    tokenizer = pkuseg.pkuseg()
elif lang == 'jp':
    tokenizer = Tokenizer()
elif lang == 'kr':
    tokenzier = Okt()

device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(args.device_num) 
random.seed(42)

# TTS, ToneConverter
source_se = torch.load(f'{args.openvoice_dir}/checkpoints_v2/base_speakers/ses/{lang}.pth', map_location=device)
ckpt_converter = f'{args.openvoice_dir}/checkpoints_v2/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
target_se, _ = se_extractor.get_se(args.refspk_path, tone_color_converter, vad=True)
tts_model = TTS(language=lang.upper(), device=device)
default_speaker_id = list(tts_model.hps.data.spk2id.values())[0]

# tmp wav: 데이터 동시 생성 시 겹치지 않게 주의
src_path = f"tmp_{lang}_{mode}.wav"

def extract_text_from_rttm(file_path, lang, mode):
    all_texts = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Extracting text from {lang.upper()} RTTM"):
            parts = line.strip().split()
            session_id = parts[1]

            if session_id not in all_texts:
                all_texts[session_id] = []

            if mode == 'test':
                if lang == 'kr':
                    text = parts[8:]
                    text = " ".join(text)
                    text = text.split("|||")[0]
                else:
                    text = parts[10:] 
                    text = " ".join(text)
                    text = text.replace("<", "")
                    text = text.replace(">", "")
            else:
                if lang == 'kr':
                    text = parts[10:] 
                    text = " ".join(text)
                else:
                    text = parts[8:]
                    text = " ".join(text)
                    text = text.split("|||")[0]
                    text = text.replace('"', '')
                    text = text.replace("'", "")
            text = text.strip()

            all_texts[session_id].append(text)
    return all_texts

result = extract_text_from_rttm(rttm_path, lang, mode)

full_length = 0
for session_id, texts in tqdm(result.items(), total=len(result), desc=f"Processing sessions"):
    print(f"== {session_id} ==")
    os.makedirs(f"{output_dir}/{session_id}", exist_ok=True)

    i = 0
    for text in texts:
        i += 1
        # TTS (MeloTTS)
        try: 
            tts_model.tts_to_file(text, default_speaker_id, src_path, speed=1.0)
        except Exception as e:
            print(f"Error generating audio for text '{text}': {e}")
            continue
        
        try:
            tts_audio = AudioSegment.from_wav(src_path)
        except:
            print(f"Error loading audio for text '{text}': {e}")
            continue

        output_path = f"{output_dir}/{session_id}/{session_id}_{i}.wav"
        txt_path = f"{output_dir}/{session_id}/{session_id}_{i}.txt"

        # Tone converter (openvoice)
        try:
            tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
            )
            print(f"**SAVED: {output_path}")
            with open(txt_path, "w", encoding="utf-8") as f:
                if lang == 'zh':
                    words = tokenizer.cut(text)
                    text = " ".join(words)
                elif lang == 'jp':
                    words = [token.surface for token in tokenizer.tokenize(text)]
                    text = " ".join(words)
                elif lang == 'kr' and morph_flag:
                    words = tokenzier.morphs(text)
                    text = " ".join(words)
                f.write(text)

            print(f"**SAVED: {text} to {txt_path}")
        except Exception as e:
            print(f"Error converting audio for text '{text}': {e}")
            continue
        
        full_length += len(tts_audio)
        print(f"Audio length: {full_length} ms / {end_length} ms")

    if full_length > end_length:
        print(f"Full length exceeded {end_length} ms. Stopping.")
        break
