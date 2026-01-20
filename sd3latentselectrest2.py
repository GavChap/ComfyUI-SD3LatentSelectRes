import torch
import comfy.model_management


class SD3LatentSelectResV2:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        # Common aspect ratios
        ratios = [
            "1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4",
            "3:2", "2:3", "21:9", "9:21", "2:1", "1:2"
        ]

        type = ['SD3/Flux/Z-Image/Qwen/etc', 'Flux2']

        # Generate list from 1.0 to 8.0 in 0.5 steps
        # range(2, 17) corresponds to 1.0 -> 8.0 when multiplied by 0.5
        mp_list = [x * 0.5 for x in range(2, 17)]

        return {
            'required': {
                'aspect_ratio': (ratios, {"default": "1:1"}),
                'megapixels': (mp_list, {"default": 1.0}),
                'latent_type': (type, {"default": "SD3"}),
                'batch_size': ("INT", {"default": 1, "min": 1, "max": 4096})
            }
        }

    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "samples")
    FUNCTION = "return_res"
    OUTPUT_NODE = True
    CATEGORY = "generate/sd3"

    def return_res(self, aspect_ratio, megapixels, latent_type, batch_size):
        # 1. Parse the aspect ratio string
        w_ratio, h_ratio = map(int, aspect_ratio.split(':'))
        ratio_val = w_ratio / h_ratio

        # 2. Convert megapixels to total pixels (1 MP = 1,000,000 pixels)
        # Ensure megapixels is treated as a float in case it comes in as a string
        target_area = float(megapixels) * 1048576

        # 3. Calculate width and height
        # formula: height = sqrt(area / ratio)
        height = (target_area / ratio_val) ** 0.5
        width = height * ratio_val

        # 4. Round to nearest multiple of 16
        width = round(width / 16) * 16
        height = round(height / 16) * 16

        # 5. Create Latent
        # SD3 Latent channels = 16. Dimensions are 1/8th of pixel size.
        channels = 16
        divisor = 8
        if latent_type == "Flux2":
            channels = 128
            divisor = 16
        latent = torch.ones([batch_size, channels, height // divisor, width // divisor], device=self.device) * 0.0609

        return width, height, {"samples": latent}


NODE_CLASS_MAPPINGS = {
    "SD3LatentSelectResV2": SD3LatentSelectResV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3LatentSelectResV2": "SD3/Flux Calc Latent ResolutionV2"
}