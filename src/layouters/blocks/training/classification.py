from .parameter_block import ArgumentBasedParameterBlock


class ClassificationTrainingParameterBlock(ArgumentBasedParameterBlock):
    title = "Classification Parameters"

    arguments = [
        ("Learning rate", "learning_rate", (5e-5, 0, 10, 1e-5)),
        ("Weight decay", "weight_decay", (1e-5, 0, 10, 1e-5)),
        ("Epochs", "n_epochs", -1, {"info": "Number of epochs to train the model. -1 means that the model is trained until convergence as determined by early stopping."}),
        ("Batch size", "batch_size", (12, 1, None, 1), {"info": "Batch size for training. If you receive an OutOfMemoryError during training, try to reduce this value."}),
        ("Image Size", "image_size", (448, 32, 1e6, 32)),
        ("Use cutout augmentation", "use_occlusion_augmentation", False, {"info": "Designates, if cutout augmentation is used during training."}),
        ("CAM threshold", "threshold", (0.2, -1, 1, 0.1), {"info": "Discretization threshold for CAM localization/segmentation for each image."}),
        ("Normalization", "normalization", (["infer", "torch", "caffe"], "infer")),
        ("Augmented WSOL output", "enriched_wsol_output", True, {"info": "Designates, if the WSOL output is generated with several predictions over different flips and input resolutions to make the predictions more robust."}),
        ("Loss", "loss", ([
            ("Categorical Cross-Entropy", "cce"),
            ("Binary Cross-Entropy", "bce"),
        ], "cce")),
    ]


class ClassificationModuleParameterBlock(ArgumentBasedParameterBlock):
    title = "Classification Module"

    arguments = [
        ("Pooling Method", "pooling_method", ([
            ("Global Average Pooling (GAP)", "gap"),
            ("Global Max Pooling (GMP)", "gmp"),
            ("Global Log-Sum-Exp Pooling (GLSEP)", "lse"),
        ], "lse"))
    ]
