from .. import ArgumentBasedParameterBlock


class SegmentationTrainingParameterBlock(ArgumentBasedParameterBlock):
    title = "Segmentation Pre-Training Parameters"

    arguments = None

    def __init__(self, is_zeroshot, tracker_prefix=""):
        super().__init__(tracker_prefix)

        self.arguments = [
            ("Learning rate", "learning_rate", (1e-5, 0, 10, 1e-5)),
            ("Weight decay", "weight_decay", (1e-5, 0, 10, 1e-5)),
            ("Epochs", "n_epochs", 3 if is_zeroshot else -1, {"info": "Number of epochs to train the model. -1 means that the model is trained until convergence as determined by early stopping."}),
            ("Batch size", "batch_size", (12, 1, None, 1), {"info": "Batch size for training. If you receive an OutOfMemoryError during training, try to reduce this value."}),
            ("Image Size", "image_size", (448, 32, 1e6, 32)),
            ("Normalization", "normalization", (["infer", "torch", "caffe"], "infer")),
            ("Use Inverted Cutout (IC)", "use_cutout", True),
            ("IC Min", "ic_min", (32, 32, 512, 32)),
            ("IC Max", "ic_max", (224, 32, 512, 32)),
            ("IC Repetitions", "ic_reps", (2, 1, 256, 1)),
            ("Image Caching", "image_caching", True, {"info": "If set, images will be cached. Can speed up training on slow infrastructure, but potentially cause memory problems, if too many images have to be cached."}),
            ("Max. Number of Workers", "max_workers", (16, 1, 32, 1)),

            ("Loss", "loss", ([
                ("Dice", "dice"),
                ("Binary Cross-Entropy", "bce"),
                ("Binary Cross-Entropy + Dice", "bce_dice"),
            ], "bce_dice")),
        ]