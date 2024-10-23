#!/bin/bash

set -e

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
      echo "$destination_path already exists. No need to download."
  fi
}

# # download faster-whisper-large-v3
# faster_whisper_model_dir=models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478
# mkdir -p $faster_whisper_model_dir
# download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json" "$faster_whisper_model_dir/config.json"
# download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin" "$faster_whisper_model_dir/model.bin"
# download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
# download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json" "$faster_whisper_model_dir/tokenizer.json"
# download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json" "$faster_whisper_model_dir/vocabulary.json"

# # download Belle-whisper-large-v3-zh-punct-fasterwhisper
# # TODO 验证这样下载的哈希是否允许
# zh_model_dir=models/models--BELLE-2--Belle-whisper-large-v3-zh-punct/snapshots/f81f1ac2f123f118094a7baa69e532eab375600e
# mkdir -p $zh_model_dir
# download "https://huggingface.co/Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper/resolve/main/config.json" "$zh_model_dir/config.json"
# download "https://huggingface.co/Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper/resolve/main/model.bin" "$zh_model_dir/model.bin"
# download "https://huggingface.co/Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper/resolve/main/preprocessor_config.json" "$zh_model_dir/preprocessor_config.json"
# download "https://huggingface.co/Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper/resolve/main/tokenizer.json" "$zh_model_dir/tokenizer.json"
# download "https://huggingface.co/Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper/resolve/main/vocabulary.json" "$zh_model_dir/vocabulary.json"

pip install -U git+https://github.com/m-bain/whisperx.git

vad_model_dir=models/vad
mkdir -p $vad_model_dir

download $(python3 ./get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"

cog run python