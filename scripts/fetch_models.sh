#!/usr/bin/env bash
# Download model weights into models/ and check them against models.sha256.
# Auto-AVSR weights: HF mirror used by Chaplin (original release: mpc001/Visual_Speech_Recognition_for_Multiple_Languages).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/LRS3_V_WER19.1 models/lm_en_subword models/mediapipe
hf=https://huggingface.co/Amanvir
for f in model.json model.pth; do
  curl -fSL -o models/LRS3_V_WER19.1/$f "$hf/LRS3_V_WER19.1/resolve/main/$f"
  curl -fSL -o models/lm_en_subword/$f "$hf/lm_en_subword/resolve/main/$f"
done
curl -fSL -o models/mediapipe/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
(cd models && sha256sum -c ../models.sha256)
