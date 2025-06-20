from ..training.parameter_block import ArgumentBasedParameterBlock


class CoverInferenceParameterBlock(ArgumentBasedParameterBlock):
    title = "Cover-Trained Inference"
    arguments = [
        ("Use Custom Image Size", "use_custom_img_size", False, {"info": "If set, a provided image size will be used during inference, otherwise a pre-set image size for the model (e.g., the training size) will be used."}),
        ("Image Size (H W)", "image_size", [
            lambda w, h: str(h) + " " + str(w),
            ("Image Width", "image_width", (3200, 0, None, 32)),
            ("Image Height", "image_height", (1600, 0, None, 32)),
        ]),
    ]