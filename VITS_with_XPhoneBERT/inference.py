import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
from memory_profiler import profile

import os
os.environ["TORCH_CUDNN_V8_API_DISABLED"]="1"    
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from scipy.io.wavfile import write
import soundfile as sf
from text2phonemesequence import Text2PhonemeSequence
from tqdm import tqdm

from text.cleaners import english_cleaners2
get_espeak_phonemes = english_cleaners2
'''
phonemizer_backend = initialize_phonemizer(punctuation_marks='-;:,.!?¡¿—…\'"«»“”(){}')
def get_espeak_phonemes(text):
    phones = phonemize(text, 
                       phonemizer=phonemizer_backend,
                       language='en-us', backend='espeak', strip=True, 
                       preserve_punctuation=True, with_stress=True, njobs=1, 
                       language_switch='remove-flags',
                        punctuation_marks='-;:,.!?¡¿—…\'"«»“”(){}')
    phones = phones.replace(" ", "▁")
    phones = " ".join(phones).replace("ˈ ", "ˈ").replace("ˌ ", "ˌ").replace(" ː", "ː")
    phones = phones.replace("ˈɚ", "ɚ").replace("ᵻ","ɪ")
    return phones
'''

@profile
def get_inputs(text, model, tokenizer_xphonebert):
    with torch.no_grad():
        # phones = model.infer_sentence(text)
        phones = get_espeak_phonemes(text)
        tokenized_text = tokenizer_xphonebert(phones)
        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        input_ids = torch.LongTensor(input_ids).cuda()
        attention_mask = torch.LongTensor(attention_mask).cuda()
        return input_ids, attention_mask

hps = utils.get_hparams_from_file("configs/lj_base_xphonebert.json")
tokenizer_xphonebert = AutoTokenizer.from_pretrained(hps.bert)
# Load Text2PhonemeSequence
model = Text2PhonemeSequence(language='eng-us', is_cuda=True)
net_g = SynthesizerTrn(
    hps.bert,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

#_ = utils.load_checkpoint("./logs/lj_base_xphonebert/G_161200.pth", net_g, None)
_ = utils.load_checkpoint("/home/latent/Downloads/pretrained_ljs.pth", net_g, None)

# f = open('/home/latent/XPhoneBERT/VITS_with_XPhoneBERT/filelists/ljs_audio_text_test_filelist_preprocessed.txt', 'r')
with open('demo_text.txt') as f:
    list_lines = f.readlines()
    f.close()
silence_duration= 22050

i = 0
@profile
def infer_text(list_lines):
    global i
    final_audio = np.array([])
    # for i, line in enumerate(tqdm(list_lines)):
        # line = line.strip().split('|')
        # assert len(line) == 2
    #stn_tst, attention_mask = get_inputs(phones, tokenizer_xphonebert, cuda_device_number)
    for line in list_lines:
        stn_tst, attention_mask = get_inputs(line, model, tokenizer_xphonebert)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            attention_mask = attention_mask.cuda().unsqueeze(0)
            audio = net_g.infer(x_tst, attention_mask, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            final_audio = np.concatenate((final_audio, audio, np.zeros(silence_duration)))
            sent_audio_path = f"output_{i}.wav"
            # sf.write(sent_audio_path, np.concatenate((audio, np.zeros(silence_duration))), 22050)
            # del final_audio
        i+=1
        print(i)

    #ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

print(len(list_lines))
input()
for _ in range(100):
    infer_text(list_lines)