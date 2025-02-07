import tensorflow as tf

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from datetime import datetime
import noisereduce as nr

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor
print(tf.__version__)

processor = AutoProcessor.from_pretrained(pretrained_path="./models/georgian_mapper.json")

fs_config = AutoConfig.from_pretrained('./examples/tacotron2/conf/tacotron2.georgian.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=fs_config,
    pretrained_path="./models/model-220000.h5"
)
tacotron2.setup_window(win_front=10, win_back=10)

melgan_config = AutoConfig.from_pretrained('./examples/multiband_melgan/conf/multiband_melgan.v1.yaml')
melgan = TFAutoModel.from_pretrained(
    config=melgan_config,
    pretrained_path="./models/generator-840000.h5"
)

def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
  input_ids = processor.text_to_sequence(input_text)
  print('ids:')
  print(input_ids)

  # text2mel part
  if text2mel_name == "TACOTRON":
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
  elif text2mel_name == "FASTSPEECH":
    mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  elif text2mel_name == "FASTSPEECH2":
    mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  else:
    raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

  # vocoder part
  if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  elif vocoder_name == "MB-MELGAN":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  else:
    raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

  if text2mel_name == "TACOTRON":
    return audio.numpy()
  else:
    return audio.numpy()

def split_into_sentences(text):
  from nltk.tokenize import sent_tokenize
  sentences = sent_tokenize(text)
  return sentences

def generate_wav_filename():
    # Get the current date and time
    current_time = datetime.now()
    # Format the date and time as a string
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    # Create the filename with the .wav extension
    filename = f"audio_{formatted_time}.wav"
    return filename

def get_wav(input_text):
    sentences = split_into_sentences(input_text)
    merged_audio = []
    for i, sentence in enumerate(sentences):
        audios = do_synthesis(sentence, tacotron2, melgan, "TACOTRON", "MB-MELGAN")
        if i == 0:
            merged_audio = audios
        else:
            merged_audio = np.concatenate((merged_audio, audios))        
          
    output_filename = generate_wav_filename()
    #reduced_noise = nr.reduce_noise(y=merged_audio, sr=22050)
    #write('./static/' + output_filename, 22050, reduced_noise)
    write('./static/' + output_filename, 22050, merged_audio)
    return output_filename

