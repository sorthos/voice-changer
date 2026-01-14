import torch
from torch import device
from voice_changer.RVC.embedder.Embedder import Embedder
from fairseq import checkpoint_utils


class FairseqHubert(Embedder):
    def loadModel(self, file: str, dev: device, isHalf: bool = True) -> Embedder:
        super().setProps("hubert_base", file, dev, isHalf)

        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [file],
            suffix="",
        )
        model = models[0]
        model.eval()

        # IMPORTANT: Fairseq HuBERT doesn't work correctly on MPS (produces garbage features).
        # Force CPU execution for the embedder on MPS devices.
        # The RVC inferencer can still use MPS for the actual voice conversion.
        if dev.type == "mps":
            print("[FairseqHubert] MPS detected - forcing CPU for HuBERT (known MPS compatibility issue)", flush=True)
            self._use_cpu_for_inference = True
            model = model.to(torch.device("cpu"))
        else:
            self._use_cpu_for_inference = False
            model = model.to(dev)
            if isHalf:
                model = model.half()

        self.model = model
        return self

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        # For MPS devices, run embedder on CPU then move result back
        if getattr(self, '_use_cpu_for_inference', False):
            inference_dev = torch.device("cpu")
            output_dev = self.dev  # Original MPS device
        else:
            inference_dev = self.dev
            output_dev = self.dev

        padding_mask = torch.BoolTensor(feats.shape).to(inference_dev).fill_(False)

        # オリジナル_v1は L9にfinal_projをかけていた。(-> 256)
        # オリジナル_v2は L12にfinal_projをかけない。(-> 768)

        inputs = {
            "source": feats.to(inference_dev),
            "padding_mask": padding_mask,
            "output_layer": embOutputLayer,  # 9 or 12
        }

        with torch.no_grad():
            logits = self.model.extract_features(**inputs)
            if useFinalProj:
                feats = self.model.final_proj(logits[0])
            else:
                feats = logits[0]

        # Move features back to original device (MPS) for downstream processing
        if output_dev != inference_dev:
            feats = feats.to(output_dev)

        return feats
