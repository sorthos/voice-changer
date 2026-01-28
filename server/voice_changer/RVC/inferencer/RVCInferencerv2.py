import torch
from const import EnumInferenceTypes
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
from .rvc_models.infer_pack.models import SynthesizerTrnMs768NSFsid


class RVCInferencerv2(Inferencer):
    def loadModel(self, file: str, gpu: int):
        self.setProps(EnumInferenceTypes.pyTorchRVCv2, file, True, gpu)

        dev = DeviceManager.get_instance().getDevice(gpu)
        isHalf = DeviceManager.get_instance().halfPrecisionAvailable(gpu)

        # IMPORTANT: RVC model doesn't work correctly on MPS (produces garbage audio).
        # Force CPU execution on Mac until MPS compatibility is resolved.
        if dev.type == "mps":
            print("[RVCInferencerv2] MPS detected - forcing CPU for RVC inference (known MPS compatibility issue)", flush=True)
            dev = torch.device("cpu")
            isHalf = False  # CPU doesn't benefit from half precision

        cpt = torch.load(file, map_location="cpu")
        model = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=isHalf)

        model.eval()
        model.load_state_dict(cpt["weight"], strict=False)

        model = model.to(dev)
        if isHalf:
            model = model.half()

        self.model = model
        self._device = dev  # Store for inference
        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        convert_length: int | None,
    ) -> torch.Tensor:
        # Ensure inputs are on the correct device
        dev = getattr(self, '_device', feats.device)
        feats = feats.to(dev)
        pitch_length = pitch_length.to(dev)
        pitch = pitch.to(dev)
        pitchf = pitchf.to(dev)
        sid = sid.to(dev)

        res = self.model.infer(feats, pitch_length, pitch, pitchf, sid, convert_length=convert_length)
        res = res[0][0, 0].to(dtype=torch.float32)
        res = torch.clip(res, -1.0, 1.0)
        return res        

