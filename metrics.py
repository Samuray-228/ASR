import argparse
import pathlib
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def calc_metrics(pred_dir, transcriptions_dir):
    pred_dir = Path(pred_dir)
    transcriptions_dir = Path(transcriptions_dir)

    cer = 0
    wer = 0
    n = 0
    target = None
    predicted = None
    
    for file in pred_dir.iterdir():
        filename = file.name
        
        with open(transcriptions_dir / filename, 'r') as f_gt:
            target = f_gt.read()
        target = CTCTextEncoder.normalize_text(target)
        
        with open(file, 'r') as f_pred:
            predicted = f_pred.read()

        cer += calc_cer(target, predicted)
        wer += calc_wer(target, predicted)
        n += 1

    print(f"CER: {cer / n}")
    print(f"WER: {wer / n}")


if __name__ == "__main__":
    calc_metrics(pred_dir="/content/ASR/data/saved/preds/test", transcriptions_dir="/content/ASR/data/transcriptions")
