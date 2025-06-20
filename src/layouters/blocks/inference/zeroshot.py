from ..training.parameter_block import ArgumentBasedParameterBlock


class ZeroshotInferenceParameterBlock(ArgumentBasedParameterBlock):
    title = "Zero-Shot Inference"
    #arg_handler = cap.arguments.zeroshot_arg_handler
    arguments = [
        ("Use Custom Image Size", "use_custom_img_size", False),
        ("Image Size (H W)", "image_size", [
            lambda w, h: str(h) + " " + str(w),
            ("Image Width", "image_width", (2688, 0, None, 32)),
            ("Image Height", "image_height", (1536, 0, None, 32)),
        ]),
        #("Normalization", "normalization", (["infer", "torch", "caffe"], "infer")),
        #("Augmented Prediction", "enriched_eval", False),
        # Enrichment params are default
        # Interlace size is default
        #("Max. Number of Workers", "max_workers", (16, 1, 32, 1)),
    ]