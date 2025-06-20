from src.layouters import TrainingLayout
from src.utils import ComponentValueTracker

normalizations = {
    "resnet_keraslike.pth": "caffe",
    "convnext_tiny": "torch",
    "convnext_small": "torch",
    "convnext_base": "torch",
    "convnext_large": "torch",
    "efficientnet_v2_s": "torch",
    "efficientnet_v2_m": "torch",
    "efficientnet_v2_l": "torch",
}

class TrainingEngine:
    def __init__(self, cover_training):
        self.cover_training = cover_training

        self.tracker = ComponentValueTracker()

    def training(self, *params):
        params = ComponentValueTracker.tracked_list_to_dict(
            self.tracker,
            params,
        )

        params["zeroshot"] = not self.cover_training

        for k, v in params.items():
            if isinstance(v, list):
                params[k] = " ".join(str(it) for it in v)

        print(params)

        prefixes = ("c_", "s_")

        if self.cover_training:
            prefixes += ("cp_",)

        # Normalization inference
        for prefix in prefixes:
            if params[prefix + "normalization"] == "infer":
                if prefix == "c_":
                    params[prefix + "normalization"] = normalizations[params["wf_base_network"]]
                else:
                    params[prefix + "normalization"] = normalizations[params["sf_base_network"]]

        cmd = "python construct_ensemble.py \\\n" + " \\\n".join([f"\t--{k} {v}" for k, v in params.items()])

        return cmd



    def training_layout(self, *args, **kwargs):
        TrainingLayout(
            with_cover_training=self.cover_training,
        )(
            self.training,
            tracker=self.tracker,
            *args,
            **kwargs,
        )