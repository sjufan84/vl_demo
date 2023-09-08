from typing import Dict, List, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import base64
from io import BytesIO
import soundfile as sf

class EndpointHandler:
    def __init__(self, path=""):
        # load model and processor from path
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
        self.model.generation_config.max_new_tokens = 650
    def __call__(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Args:
            data (:dict:):
                The payload with the text prompt and generation parameters.
        """
        # process input
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)
        audio_base64 = data.pop("audio_data", None)
        
        # preprocess
        input_data = {
            'text': [inputs],
            'padding': True,
            'return_tensors': "pt",
        }
       
        if audio_base64:
            audio_data = base64.b64decode(audio_base64)
            audio_buffer = BytesIO(audio_data)
            audio_data, sampling_rate = sf.read(audio_buffer)
            audio_array = audio_data[: len(audio_data) // 4]
            input_data['audio'] = audio_array
            input_data['sampling_rate'] = 32000 # Or a fixed rate if you know it

        inputs = self.processor(**input_data).to("cuda")

        # pass inputs with all kwargs in data
        with torch.autocast("cuda"):
            if parameters is not None:
                outputs = self.model.generate(**inputs, **parameters)
            else:
                outputs = self.model.generate(**inputs)
        
        prediction = outputs[0].cpu().numpy().tolist()

        return [{"generated_audio": prediction}]