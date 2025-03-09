import sys 
import os
os.environ["PYTHONPATH"] = "third_party/Matcha-TTS"
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import json
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
from pathlib import Path
import random
import whisper
import datetime  # 用于生成时间戳

from evaluation.speaker_similarity_resemblyzer import extract_similarity
from evaluation.f0_pearson_coefficients import extract_fpc
from evaluation.mel_cepstral_distortion import extract_mcd

max_val = 0.8

def split_tts_text(text_content, max_length=100):
    segments = []
    tmp_str = text_content.strip()
    while tmp_str:
        if len(tmp_str) <= max_length:
            segments.append(tmp_str)
            break
        split_index = max_length
        while split_index > 0 and tmp_str[split_index] not in (' ', ',', '.', '!', '?', '，', '；', ':', '。', '！', '？', '、', '：'):
            split_index -= 1
        if split_index == 0:
            split_index = max_length
        segments.append(tmp_str[:split_index].strip())
        tmp_str = tmp_str[split_index:].strip()
    return segments

def generate_tts_audio(tts_text, instruct, prompt_text, prompt_speech, cosyvoice):
    """
    使用 inference_instruct2 生成 TTS 音频，要求 instruct 非空
    """
    if not instruct.strip():
        raise ValueError("自然语言控制指令 (instruct) 不能为空！请提供有效的控制指令。")
    audio_segments = []
    text_segments = split_tts_text(tts_text)
    start_time = time.time()
    for i, segment in enumerate(text_segments):
        for j, result in enumerate(cosyvoice.inference_instruct2(
            segment,
            instruct, 
            prompt_speech,
            speed=1,
            stream=False
        )):
            audio = result['tts_speech']
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio_segments.append(audio)
    complete_audio = torch.cat(audio_segments, dim=1)
    generation_time = time.time() - start_time
    return complete_audio, generation_time

def recognize_audio_whisper(audio_path):
    """
    使用 Whisper large-v3 模型对音频进行识别，返回识别文本。
    """
    model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path)
    recognized_text = result["text"].strip()
    return recognized_text

def apply_natural_language_control_to_text(tts_text, instruct):
    """
    在 TTS 文本上添加自然语言控制指令：
    如果 instruct 为空则不修改；
    否则要求 instruct 必须在预定义模板中：
        ["平静", "开心", "悲伤", "愤怒", "惊喜", "凶猛", "好奇", "优雅", "孤独", "威严",
         "焦虑", "恐惧", "羞耻", "嫉妒", "感激", "冷漠", "兴奋", "困惑", "骄傲", "愧疚", "失望", "幸福", "无助", "爱慕", "厌恶", "渴望", "疲惫", "坚定", "脆弱", "温柔", "英雄", "反派", "智者", "顽童", "守护者", "探险家", "隐士", "领袖", "追随者", "导师", "学徒", "叛逆者", "艺术家", "科学家", "战士", "间谍", "贵族", "平民", "魔法师", "预言家"]
    如果符合要求，则在文本末尾追加【自然语言控制指令：instruct】；
    否则报错退出。
    """
    allowed_instruct = [
        "用开心的语气说", "用伤心的语气说", "用惊讶的语气说", "用恐惧的情感表达", "用恶心的情感表达", 
        "冷静", "严肃", "快速", "非常快速", "慢速", "非常慢速", 
        "神秘", "凶猛", "好奇", "优雅", "孤独"
        ]
    if not instruct.strip():
        print("未应用自然语言控制")
        return tts_text
    if instruct.strip() not in allowed_instruct:
        print(f"错误：提供的自然语言控制指令 '{instruct.strip()}' 不在允许列表中，请选择以下之一：{allowed_instruct}")
        exit(1)
    print(f"使用自然语言控制指令： {instruct.strip()}")
    return tts_text

def apply_fine_grained_control_to_text(tts_text, custom_control=''):
    """
    如果用户希望手动在文本中插入细粒度控制标记，则直接返回原始 TTS 文本。
    用户只能选择下面预定义的 token，否则随机选择一个：
    Allowed tokens: ['[breath]', '<strong>', '</strong>', '[noise]',
                     '[laughter]', '[cough]', '[clucking]', '[accent]',
                     '[quick_breath]', '<laughter>', '</laughter>',
                     '[hissing]', '[sigh]', '[vocalized-noise]',
                     '[lipsmack]', '[mn]']
    """
    allowed_tokens = ['[breath]', '<strong>', '</strong>', '[noise]',
                      '[laughter]', '[cough]', '[clucking]', '[accent]',
                      '[quick_breath]', '<laughter>', '</laughter>',
                      '[hissing]', '[sigh]', '[vocalized-noise]',
                      '[lipsmack]', '[mn]']
    token = custom_control.strip()
    if not token:
        print("未应用细粒度控制")
        return tts_text
    if token not in allowed_tokens:
        print(f"警告：提供的细粒度控制标记 '{token}' 不在允许列表中，将随机选择一个。")
        token = random.choice(allowed_tokens)
        print(f"随机选择的细粒度控制标记: {token}")
    # 返回原始文本，由用户自行在合适位置插入该 token
    return tts_text

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_path, instruct,
                   use_natural_language=False, use_fine_control=False, custom_fine_control=''):
    # 判断是否使用自然语言控制
    use_instruct = use_natural_language or instruct.strip()

    if mode_checkbox_group == 'Generate Speech Using Pre-trained Voice':
        # 使用预训练音色生成语音模式：使用预定义音色字典（此时 sft 有意义）
        speaker_key = sft_dropdown
        prompt_config = {
            "en_man1": {
                "audio_path": "./asset/Philippines_prompt/en_man1_prompt.wav",
                "prompt_text": "My greetings to the participants and organizers of the 50th Plenary Session of the Committee on World Food Security."
            },
            "en_woman1": {
                "audio_path": "./asset/Philippines_prompt/en_woman1_prompt.wav",
                "prompt_text": "really makes a great impact on the rhetoric that this is the kind of environment that we are encouraging."
            },
            "en_man2": {
                "audio_path": "./asset/Philippines_prompt/en_man2_prompt.wav",
                "prompt_text": "So I said, if we do not end the dick, this problem, the next generation will be having a serious problem."
            },
            "en_woman2": {
                "audio_path": "./asset/Philippines_prompt/en_woman2_prompt.wav",
                "prompt_text": "Today, the Philippines lost a national treasure. Cori Aquino helped lead a revolution that restored democracy and rule of law to our nation at a time of great peril. Our nation will mourn her passing."
            },
            "zh_man": {
                "audio_path": "./asset/Philippines_prompt/zh_man_prompt.wav",
                "prompt_text": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
            },
            "zh_woman": {
                "audio_path": "./asset/Philippines_prompt/zh_woman_prompt.wav",
                "prompt_text": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
            }
        }
        if speaker_key in prompt_config:
            speaker_config = prompt_config[speaker_key]
            prompt_speech = load_wav(speaker_config["audio_path"], 16000)
            prompt_text = speaker_config["prompt_text"]
        else:
            logging.warning("未找到指定的预训练音色，使用默认音色")
            speaker_key = "en_man1"
            speaker_config = prompt_config[speaker_key]
            prompt_speech = load_wav(speaker_config["audio_path"], 16000)
            prompt_text = speaker_config["prompt_text"]

        print(f"[Debug] Selected pre-trained voice key: {speaker_key}")
        print(f"[Debug] Prompt audio path: {speaker_config['audio_path']}")
        print(f"[Debug] Prompt text: {prompt_text}")
        print(f"[Debug] Loaded prompt audio sample rate: 16000")
        print(f"[Debug] Original TTS text: {tts_text}")

        modified_tts_text = tts_text
        if use_natural_language:
            modified_tts_text = apply_natural_language_control_to_text(modified_tts_text, instruct)
        if use_fine_control:
            modified_tts_text = apply_fine_grained_control_to_text(modified_tts_text, custom_fine_control)
        print(f"[Debug] Modified TTS text: {modified_tts_text}")
        print(f"[Debug] Natural language control instruction: {instruct}")

        if use_instruct:
            print("[Debug] Generating speech using natural language control")
            print(f"[Debug] Uing control:{instruct}")
            complete_audio, generation_time = generate_tts_audio(
                modified_tts_text,
                instruct,
                prompt_text,
                prompt_speech,
                cosyvoice
            )
            print(f"[Debug] Generation time: {generation_time:.2f} seconds")
            return complete_audio
        else:
            print("[Debug] Generating speech using zero_shot")
            audio_segments = []
            text_segments = split_tts_text(modified_tts_text)
            start_time = time.time()
            for segment in text_segments:
                for result in cosyvoice.inference_zero_shot(
                    segment,
                    prompt_text,
                    prompt_speech,
                    stream=False
                ):
                    audio = result['tts_speech']
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    audio_segments.append(audio)
            complete_audio = torch.cat(audio_segments, dim=1)
            generation_time = time.time() - start_time
            print(f"[Debug] Generation time: {generation_time:.2f} seconds")
            return complete_audio

    elif mode_checkbox_group == 'Voice Cloning':
        # Voice Cloning mode: 仅支持用户上传参考音频，不支持录音模式
        if prompt_wav_path is not None and os.path.exists(prompt_wav_path):
            prompt_speech = load_wav(prompt_wav_path, 16000)
            print(f"[Debug] Loaded uploaded reference audio: {prompt_wav_path}, sample rate: 16000")
        else:
            logging.error("Voice Cloning mode requires a reference audio file upload")
            return None

        recognized_text = recognize_audio_whisper(prompt_wav_path)
        print(f"Automatically recognized text: {recognized_text}")
        user_input = input("Please confirm the recognized text (press Enter to accept, or input the correct text): ").strip()
        if user_input:
            prompt_text = user_input
        else:
            prompt_text = recognized_text

        modified_tts_text = tts_text
        if use_natural_language:
            modified_tts_text = apply_natural_language_control_to_text(modified_tts_text, instruct)
        if use_fine_control:
            modified_tts_text = apply_fine_grained_control_to_text(modified_tts_text, custom_fine_control)
        print(f"[Debug] Modified TTS text: {modified_tts_text}")
        print(f"[Debug] Natural language control instruction: {instruct}")

        audio_segments = []
        text_segments = split_tts_text(modified_tts_text)
        start_time = time.time()
        if use_instruct:
            print("[Debug] Generating speech using natural language control (Voice Cloning mode)")
            for segment in text_segments:
                for result in cosyvoice.inference_instruct2(
                    segment,
                    instruct,
                    prompt_speech,
                    speed=1,
                    stream=False
                ):
                    audio = result['tts_speech']
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    audio_segments.append(audio)
        else:
            print("[Debug] Generating speech using zero_shot (Voice Cloning mode)")
            for segment in text_segments:
                for result in cosyvoice.inference_zero_shot(
                    segment,
                    prompt_text,
                    prompt_speech,
                    stream=False
                ):
                    audio = result['tts_speech']
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    audio_segments.append(audio)
        complete_audio = torch.cat(audio_segments, dim=1)
        generation_time = time.time() - start_time
        print(f"[Debug] Generation time: {generation_time:.2f} seconds")
        return complete_audio

    else:
        logging.warning("Unknown mode, defaulting to Generate Speech Using Pre-trained Voice")
        speaker_key = sft_dropdown
        prompt_config = {
            "en_man1": {
                "audio_path": "./asset/Philippines_prompt/en_man1_prompt.wav",
                "prompt_text": "My greetings to the participants and organizers of the 50th Plenary Session of the Committee on World Food Security."
            },
            "en_woman1": {
                "audio_path": "./asset/Philippines_prompt/en_woman1_prompt.wav",
                "prompt_text": "really makes a great impact on the rhetoric that this is the kind of environment that we are encouraging."
            },
            "en_man2": {
                "audio_path": "./asset/Philippines_prompt/en_man2_prompt.wav",
                "prompt_text": "So I said, if we do not end the dick, this problem, the next generation will be having a serious problem."
            },
            "en_woman2": {
                "audio_path": "./asset/Philippines_prompt/en_woman2_prompt.wav",
                "prompt_text": "Today, the Philippines lost a national treasure. Cori Aquino helped lead a revolution that restored democracy and rule of law to our nation at a time of great peril. Our nation will mourn her passing."
            },
            "zh_man": {
                "audio_path": "./asset/Philippines_prompt/zh_man_prompt.wav",
                "prompt_text": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
            },
            "zh_woman": {
                "audio_path": "./asset/Philippines_prompt/zh_woman_prompt.wav",
                "prompt_text": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？"
            }
        }
        if speaker_key in prompt_config:
            speaker_config = prompt_config[speaker_key]
            prompt_speech = load_wav(speaker_config["audio_path"], 16000)
            prompt_text = speaker_config["prompt_text"]
        else:
            logging.warning("未找到指定的预训练音色，使用默认音色")
            speaker_key = "zh_man"
            speaker_config = prompt_config[speaker_key]
            prompt_speech = load_wav(speaker_config["audio_path"], 16000)
            prompt_text = speaker_config["prompt_text"]
        modified_tts_text = tts_text
        if use_natural_language:
            modified_tts_text = apply_natural_language_control_to_text(modified_tts_text, instruct)
        if use_fine_control:
            modified_tts_text = apply_fine_grained_control_to_text(modified_tts_text, custom_fine_control)
        if use_instruct:
            complete_audio, generation_time = generate_tts_audio(
                modified_tts_text,
                instruct,
                prompt_text,
                prompt_speech,
                cosyvoice
            )
        else:
            audio_segments = []
            text_segments = split_tts_text(modified_tts_text)
            start_time = time.time()
            for segment in text_segments:
                for result in cosyvoice.inference_zero_shot(
                    segment,
                    prompt_text,
                    prompt_speech,
                    stream=False
                ):
                    audio = result['tts_speech']
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                    audio_segments.append(audio)
            complete_audio = torch.cat(audio_segments, dim=1)
            generation_time = time.time() - start_time
        return complete_audio

def plot_spectrogram(audio_path, title, output_image_path, sr=24000):
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def main():
    logging.disable(logging.DEBUG)  # Disable DEBUG and lower level logs
    logging.getLogger('modelscope').setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice2-0.5B', help='Local path or modelscope repo id')
    parser.add_argument('--text', type=str, required=True, help='TTS text')
    parser.add_argument('--mode', type=str, choices=['Generate Speech Using Pre-trained Voice', 'Voice Cloning'], required=True, help='Inference mode')
    parser.add_argument('--sft', type=str, default='zh_man', help='Used in Pre-trained Voice mode; ignored in Voice Cloning mode')
    parser.add_argument('--prompt_text', type=str, default='', help='Prompt text for cloning')
    parser.add_argument('--prompt_wav', type=str, default=None, help='Path to prompt audio file (for Voice Cloning mode)')
    parser.add_argument('--use_natural_language', action='store_true', help='Enable natural language control')
    parser.add_argument('--instruct', type=str, default='', help='Instruction text for synthesis (must be one of ["平静", "开心", "悲伤", "愤怒", "惊喜", "凶猛", "好奇", "优雅", "孤独", "威严", "焦虑", "恐惧", "羞耻", "嫉妒", "感激", "冷漠", "兴奋", "困惑", "骄傲", "愧疚", "失望", "幸福", "无助", "爱慕", "厌恶", "渴望", "疲惫", "坚定", "脆弱", "温柔", "英雄", "反派", "智者", "顽童", "守护者", "探险家", "隐士", "领袖", "追随者", "导师", "学徒", "叛逆者", "艺术家", "科学家", "战士", "间谍", "贵族", "平民", "魔法师", "预言家"]')
    parser.add_argument('--use_fine_control', action='store_true', help='Enable fine-grained control')
    parser.add_argument('--custom_fine_control', type=str, default='', help='Custom fine-grained control marker (choose from allowed tokens)')
    parser.add_argument('--filename', type=str, default=None, help='Custom file name prefix for output files')
    parser.add_argument('--base_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--task_id', type=str, default='task1', help='Task ID')
    args = parser.parse_args()
    
    try:
        global cosyvoice
        cosyvoice = CosyVoice2(args.model_dir)
    except Exception:
        raise TypeError('No valid model_type!')
    
    # 根据用户是否提供 --filename 参数决定文件名前缀
    if args.filename is not None:
        filename_prefix = args.filename
    else:
        if args.mode == 'Voice Cloning':
            filename_prefix = f"{args.task_id}-upload-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            filename_prefix = f"{args.sft}-{args.instruct if args.instruct.strip() else 'noControl'}-{args.task_id}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    audio_output = generate_audio(
        args.text,
        args.mode,
        args.sft,
        args.prompt_text,
        args.prompt_wav,
        args.instruct,
        use_natural_language=args.use_natural_language,
        use_fine_control=args.use_fine_control,
        custom_fine_control=args.custom_fine_control
    )
    
    if audio_output is not None:
        output_wav_path = os.path.join(args.base_dir, f"{filename_prefix}.wav")
        torchaudio.save(output_wav_path, audio_output, cosyvoice.sample_rate)
        print(f"Saved generated audio to: {output_wav_path}")
    
        gen_spec_path = os.path.join(args.base_dir, f"{filename_prefix}_spectrogram.png")
        plot_spectrogram(output_wav_path, "Spectrogram of Generated Audio", gen_spec_path)
        print(f"Saved spectrogram to: {gen_spec_path}")
    
        if args.mode == 'Voice Cloning':
            ref_audio_path = args.prompt_wav
        else:
            prompt_config = {
                "en_man1": "./asset/Philippines_prompt/en_man1_prompt.wav",
                "en_woman1": "./asset/Philippines_prompt/en_woman1_prompt.wav",
                "en_man2": "./asset/Philippines_prompt/en_man2_prompt.wav",
                "en_woman2": "./asset/Philippines_prompt/en_woman2_prompt.wav",
                "zh_man": "./asset/Philippines_prompt/zh_man_prompt.wav",
                "zh_woman": "./asset/Philippines_prompt/zh_woman_prompt.wav"
            }
            if args.sft in prompt_config:
                ref_audio_path = prompt_config[args.sft]
            else:
                ref_audio_path = prompt_config["zh_man"]
    
        metrics_kwargs = {"kwargs": {"fs": cosyvoice.sample_rate, "method": "cut", "need_mean": True}}
        mean_similarity = extract_similarity(ref_audio_path, output_wav_path, similarity_mode="pairwith")
        mcd_val = extract_mcd(ref_audio_path, output_wav_path, **metrics_kwargs) if ref_audio_path else None
    
        metrics = {
            "Speaker_Similarity": float(mean_similarity),
            "Mel_Cepstral_Distortion": float(mcd_val) if mcd_val is not None else None,
        }
    
        metrics_file = os.path.join(args.base_dir, f"{filename_prefix}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Voice generated successfully! Metrics saved to: {metrics_file}")
    else:
        print("Voice generation failed. Please check logs.")

if __name__ == '__main__':
    main()
