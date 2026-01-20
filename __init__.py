from .sd3latentselectres import SD3LatentSelectRes
from .sd3latentselectrest2 import SD3LatentSelectResV2

NODE_CLASS_MAPPINGS = {
    "SD3LatentSelectRes": SD3LatentSelectRes,
    "SD3LatentSelectResV2": SD3LatentSelectResV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SD3LatentSelectRes": "SD3/Flux Select Latent Resolution",
    "SD3LatentSelectResV2": "SD3/Flux Select Latent Resolution V2"
}
