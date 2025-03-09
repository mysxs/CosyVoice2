# CosyVoice2

## 整体逻辑
该脚本实现了基于 CosyVoice2 模型的文本到语音合成，支持两种合成模式：
### Generate Speech Using Pre-trained Voice
- 根据命令行参数 --sft 指定的预训练音色标识，从预定义的提示音频和提示文本字典中选取对应内容。
- 将用户输入的 TTS 文本作为基础，如果用户通过参数 --use_natural_language 启用了自然语言控制，则调用函数 apply_natural_language_control_to_text。
- 在该函数中，如果用户在 --instruct 中提供了控制文本，则程序会检查该文本是否在预定义模板列表中（允许的风格为：["平静", "开心", "悲伤", "愤怒", "惊喜", "凶猛", "好奇", "优雅", "孤独", "威严"]）。
- 如果符合要求，则在原 TTS 文本末尾追加【自然语言控制指令：xxx】；如果不符合，则程序报错退出。
- 如果用户同时启用了细粒度控制（通过 --use_fine_control 并传入 --custom_fine_control），则调用细粒度控制函数，要求用户提供的 token 必须从允许列表中选择（例如 [breath]、[laughter]、[noise] 等）。
- 根据是否启用自然语言控制（由 --use_natural_language 和 --instruct 判断），调用对应的接口生成语音（若启用，则调用 cosyvoice.inference_instruct2，否则调用 cosyvoice.inference_zero_shot）。
### Voice Cloning
 需要用户提供参考音频（通过参数 --prompt_wav）。脚本会：
-   加载该参考音频作为提示语音。
-   利用 Whisper large-v3 模型对参考音频进行自动语音识别，并打印识别结果，等待用户在终端确认或修正；
-   与预训练音色模式相似，对输入 TTS 文本根据用户是否启用自然语言控制和细粒度控制进行处理（调用相应函数），然后调用生成接口（inference_instruct2 或 inference_zero_shot）合成语音。

## 生成与输出
### 输出文件：
- 合成的语音会保存为 WAV 文件；频谱图则利用 librosa 和 matplotlib 绘制，并保存为 PNG 文件；
- 脚本还调用预设的评估指标计算函数（计算 Speaker Similarity 和 Mel Cepstral Distortion），将结果以 JSON 格式保存在指定目录中。
文件命名：
- 用户可通过 --filename 参数自定义输出文件前缀；如果未提供，则自动根据模式、预训练音色、控制指令、任务ID以及当前时间戳生成唯一文件名前缀。
