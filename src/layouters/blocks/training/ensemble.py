import gradio as gr

from .. import ArgumentBasedParameterBlock


class CoverEnsembleParameterBlock(ArgumentBasedParameterBlock):
    title = "Ensemble Parameters"


    def __init__(self, zeroshot=False, tracker_prefix=""):
        super().__init__(tracker_prefix)

        if zeroshot:
            archs = ["convnext_tiny", "convnext_base", "convnext_large"]

            ee = []
        else:
            archs = ["efficientnet_v2_l"]

            ee = [("Ensemble Epochs", "ensemble_epochs", lambda **params: gr.Dropdown(
                choices=[5, 10, 20, 40, 60],
                multiselect=True,
                allow_custom_value=True,
                value=[20],
                info="Train one model each for every provided number of epochs.",
                **params,
            ))]

        self.arguments = [
            ("Use Ensemble", "use_ensemble", False, {"info": "If set, several models will be trained according to the settings below, and aggregated into a single ensemble model to make predictions more robust. The ensemble will comprise the cartesian product of models trained with the setups specified below."}),
            *ee,
            ("Ensemble Architectures", "ensemble_models", lambda **params: gr.Dropdown(
                choices=[
                    ("ConvNext Tiny", "convnext_tiny"),
                    ("ConvNext Small", "convnext_small"),
                    ("ConvNext Base", "convnext_base"),
                    ("ConvNext Large", "convnext_large"),
                    ("EfficientNet V2 Small", "efficientnet_v2_s"),
                    ("EfficientNet V2 Medium", "efficientnet_v2_m"),
                    ("EfficientNet V2 Large", "efficientnet_v2_l"),
                ],
                multiselect=True,
                allow_custom_value=False,
                value=archs,
                info="Train one model each for every provided architecture.",
                **params,
            )),
            ("# Repetitions", "n_ensemble_reps", (1, 1, None, 1), {"info": "Adds one model of every kind to the ensemble per repetition."}),
        ]