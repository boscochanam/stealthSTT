"""Auto-AVSR visual speech model: encode mouth clips, decode text, score known phrases.

Wraps the vendored ESPnet code (src/espnet, Apache-2.0, from Auto-AVSR via Chaplin).
Three uses:
  embed(clip)          -> (T', 768) encoder frames, for template matching
  transcribe(clip)     -> n-best free text, for dictation mode
  score_phrases(clip)  -> log P(phrase | video) for each known command phrase
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece
import torch
import torchvision

ASSETS = Path(__file__).parent / "assets"

# Auto-AVSR input normalisation (pipelines/data/transforms.py VideoTransform).
_MEAN, _STD, _CROP = 0.421, 0.165, 88


@dataclass
class Hyp:
    text: str
    score: float


class VSRModel:
    def __init__(self, model_dir: Path, lm_dir: Path | None = None, device: str = "cpu",
                 beam_size: int = 10, ctc_weight: float = 0.1, lm_weight: float = 0.3,
                 threads: int | None = None):
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E

        if threads:
            torch.set_num_threads(threads)
        self.device = device
        conf = json.loads((Path(model_dir) / "model.json").read_text())
        args = argparse.Namespace(**(conf if isinstance(conf, dict) else conf[2]))
        units = (ASSETS / "unigram5000_units.txt").read_text(encoding="utf8").splitlines()
        self.unit_ids = {u.split()[0]: int(u.split()[-1]) for u in units}
        self.token_list = ["<blank>"] + list(self.unit_ids) + ["<eos>"]
        self.odim = len(self.token_list)
        self.eos = self.odim - 1
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(ASSETS / "unigram5000.model"))

        self.model = E2E(self.odim, args)
        state = torch.load(Path(model_dir) / "model.pth", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()

        self.beam_size, self.ctc_weight, self.lm_weight = beam_size, ctc_weight, lm_weight
        self.lm_dir = Path(lm_dir) if lm_dir else None
        self._beam = None  # built lazily: only dictation needs it

    # ---------- input ----------
    @staticmethod
    def prepare(clip: np.ndarray) -> torch.Tensor:
        """uint8 (T, 96, 96) grey mouth crops -> normalised (1, T, 88, 88) float tensor (C, T, H, W)."""
        x = torch.as_tensor(clip).float().unsqueeze(1) / 255.0
        x = torchvision.transforms.functional.center_crop(x, [_CROP, _CROP])
        return ((x - _MEAN) / _STD).transpose(0, 1)

    @torch.no_grad()
    def embed(self, clip: np.ndarray) -> torch.Tensor:
        return self.model.encode(self.prepare(clip).to(self.device))

    # ---------- text ----------
    def tokenize(self, text: str) -> list[int]:
        pieces = self.spm.EncodeAsPieces(text.upper())
        return [self.unit_ids.get(p, self.unit_ids["<unk>"]) for p in pieces]

    @torch.no_grad()
    def score_phrases(self, enc: torch.Tensor, phrases: list[str]) -> np.ndarray:
        """Length-normalised log-likelihood of each phrase given encoder output.

        Mixes the attention decoder (teacher-forced) and CTC, with the model's own
        ctc_weight. Returns one score per phrase; higher is better.
        """
        from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

        toks = [self.tokenize(p) for p in phrases]
        n, L = len(toks), max(len(t) for t in toks) + 1
        ys_in = torch.full((n, L), self.eos, dtype=torch.long)
        ys_out = torch.full((n, L), -1, dtype=torch.long)
        for i, t in enumerate(toks):
            ys_in[i, 1:len(t) + 1] = torch.tensor(t)
            ys_out[i, :len(t)] = torch.tensor(t)
            ys_out[i, len(t)] = self.eos
        tgt_mask = (ys_in != -1).unsqueeze(-2) & subsequent_mask(L).unsqueeze(0)
        mem = enc.unsqueeze(0).expand(n, -1, -1).to(self.device)
        logits, _ = self.model.decoder(ys_in.to(self.device), tgt_mask.to(self.device), mem, None)
        logp = logits.log_softmax(-1).cpu()
        valid = ys_out != -1
        att = (logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid).sum(1)
        att = att / valid.sum(1)

        ctc_lp = self.model.ctc.log_softmax(enc.unsqueeze(0)).squeeze(0).cpu()  # (T, V)
        T = ctc_lp.shape[0]
        ctc = torch.empty(n)
        for i, t in enumerate(toks):
            if len(t) > T:
                ctc[i] = -1e4
                continue
            loss = torch.nn.functional.ctc_loss(
                ctc_lp.unsqueeze(1), torch.tensor([t]), torch.tensor([T]), torch.tensor([len(t)]),
                blank=0, reduction="sum", zero_infinity=True)
            ctc[i] = -loss / (len(t) + 1)
        w = self.ctc_weight
        return ((1 - w) * att + w * ctc).numpy()

    def _beam_search(self):
        if self._beam is None:
            from espnet.asr.asr_utils import get_model_conf, torch_load
            from espnet.nets.batch_beam_search import BatchBeamSearch
            from espnet.nets.lm_interface import dynamic_import_lm
            from espnet.nets.scorers.length_bonus import LengthBonus

            scorers = self.model.scorers()
            lm_weight = 0.0
            if self.lm_dir and (self.lm_dir / "model.pth").exists():
                lm_args = get_model_conf(str(self.lm_dir / "model.pth"), str(self.lm_dir / "model.json"))
                lm_cls = dynamic_import_lm(getattr(lm_args, "model_module", "default"), lm_args.backend)
                lm = lm_cls(self.odim, lm_args)
                torch_load(str(self.lm_dir / "model.pth"), lm)
                scorers["lm"] = lm.eval()
                lm_weight = self.lm_weight
            scorers["length_bonus"] = LengthBonus(self.odim)
            weights = dict(decoder=1.0 - self.ctc_weight, ctc=self.ctc_weight, lm=lm_weight, length_bonus=0.0)
            self._beam = BatchBeamSearch(
                beam_size=self.beam_size, vocab_size=self.odim, weights=weights, scorers=scorers,
                sos=self.eos, eos=self.eos, token_list=self.token_list, pre_beam_score_key="decoder",
            ).to(self.device).eval()
        return self._beam

    @torch.no_grad()
    def transcribe(self, enc: torch.Tensor, nbest: int = 5) -> list[Hyp]:
        hyps = self._beam_search()(enc)
        out = []
        for h in hyps[:nbest]:
            ids = [int(i) for i in h.yseq[1:] if int(i) != self.eos]
            text = "".join(self.token_list[i] for i in ids).replace("▁", " ").strip()
            out.append(Hyp(text=text, score=float(h.score)))
        return out
