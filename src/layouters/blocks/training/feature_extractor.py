from .parameter_block import ArgumentBasedParameterBlock


class FeatureExtractorParameterBlock(ArgumentBasedParameterBlock):
    title = "Network Parameters"

    def __init__(self, kind, zeroshot=False, tracker_prefix=""):
        super().__init__(tracker_prefix)

        if kind == "wsol":
            recommended_net = "efficientnet_v2_l"
        elif kind == "segmentation":
            if zeroshot:
                recommended_net = "convnext_large"
            else:
                recommended_net = "efficientnet_v2_l"

        if zeroshot:
            recommended_layer = "P1"
            recommended_depth = 128
        else:
            recommended_layer = "P2"
            recommended_depth = 512

        self.arguments = [
            ("Architecture", "base_network", ([
                ("ConvNext Tiny", "convnext_tiny"),
                ("ConvNext Small", "convnext_small"),
                ("ConvNext Base", "convnext_base"),
                ("ConvNext Large", "convnext_large"),
                ("EfficientNet V2 Small", "efficientnet_v2_s"),
                ("EfficientNet V2 Medium", "efficientnet_v2_m"),
                ("EfficientNet V2 Large", "efficientnet_v2_l"),
            ], recommended_net)),
            ("FPN Configuration", "fpn_spec", [
                lambda l, d: f"{l}-{d}",
                ("FPN Layer", "fpn_layer", ([
                    "P1", "P2", "P3", "P4", "P5",
                ], recommended_layer)),
                ("FPN Depth", "fpn_depth", (recommended_depth, 32, None, 32))
            ])
        ]