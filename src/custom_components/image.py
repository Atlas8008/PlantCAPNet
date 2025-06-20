import PIL.Image
import numpy as np
import gradio as gr

from pathlib import Path
from gradio.data_classes import FileData
from gradio import image_utils


class Image(gr.Image):
    def postprocess(
        self, value
    ):
        """
        Parameters:
            value: Expects a `numpy.array`, `PIL.Image`, or `str` or `pathlib.Path` filepath to an image which is displayed.
        Returns:
            Returns the image as a `FileData` object.
        """
        print("Postprocessing", value)
        if value is None:
            return None
        if isinstance(value, str) and value.lower().endswith(".svg"):
            return FileData(path=value, orig_name=Path(value).name)
        if isinstance(value, str) and value.lower().endswith((".tif", ".tiff")):
            value = PIL.Image.open(value).convert("RGB")
        saved = image_utils.save_image(value, self.GRADIO_CACHE, self.format)
        orig_name = Path(saved).name if Path(saved).exists() else None
        return FileData(path=saved, orig_name=orig_name)