import os
import json
import comfy
import torch


class SD3LatentSelectRes:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        cls.size_sizes, cls.size_dict = cls.read_sizes()
        return {
            'required': {
                'size_selected': (cls.size_sizes,),
                'flip_ratio': ("BOOLEAN", {"default": False}),
                'batch_size': ("INT", {"default": 1, "min": 1, "max": 4096})
            }
        }

    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "samples")
    FUNCTION = "return_res"
    OUTPUT_NODE = True
    CATEGORY = "generate/sd3"

    def return_res(self, size_selected, flip_ratio, batch_size):
        # Extract resolution name and dimensions using the key
        selected_info = self.size_dict[size_selected]
        if (flip_ratio):
            width = int(selected_info["height"])
            height = int(selected_info["width"])    
        else:
            width = int(selected_info["width"])
            height = int(selected_info["height"])    

        latent = torch.ones([batch_size, 16, height // 8, width // 8], device=self.device) * 0.0609
        return {"width": width}, {"height": height}, {"samples": latent},

    @staticmethod
    def read_sizes():
        p = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(p, 'sizes.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
        size_sizes = [f"{key}" for key, value in data['sizes'].items()]
        size_dict = {f"{key}": value for key, value in data['sizes'].items()}
        return size_sizes, size_dict


NODE_CLASS_MAPPINGS = {
    "SD3LatentSelectRes": SD3LatentSelectRes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3LatentSelectRes": "SD3/Flux Select Latent Resolution"
}


