from .. import ArgumentBasedParameterBlock


class CoverTrainingParameterBlock(ArgumentBasedParameterBlock):
    title = "Cover Training Parameters"

    arguments = [
        ("Learning Rate (Cover/Phenology)", "learning_rate", [
            lambda lr1, lr2: str(lr1) + " " + str(lr2),
            ("Cover Learning Rate", "cover_lr", (1e-5, 0, None, 1e-5)),
            ("Phenology Learning Rate", "pheno_lr", (1e-5, 0, None, 1e-5)),
        ]),
        ("# Epochs", "n_epochs", (30, 1, None, 1)),
        ("Weight Decay", "weight_decay", (0, 0, None, 1e-5)),
        ("Dataset Mode", "dataset_mode", "daily"),
        ("Phenology Model", "pheno_model", ([
            ("None", "none"),
            ("Flowering/Senescence", "fs"),
        ], "fs")),
        ("Image Size (H W)", "image_size", [
            lambda w, h: str(h) + " " + str(w),
            ("Image Width", "image_width", (2688, 32, None, 32)),
            ("Image Height", "image_height", (1536, 32, None, 32)),
        ]),
        ("Cover Loss", "loss", ([
            ("Hellinger Distance", "hell"),
            ("Mean Absolute Error", "mae"),
            ("Mean Squared Error", "mse"),
        ], "hell")),
        ("Phenology Loss", "pheno_loss", ([
            ("Mean Absolute Error", "mae"),
            ("Dice", "dice"),
        ], "mae")),
        ("Normalization", "normalization", (["infer", "torch", "caffe"], "infer")),
        # Output folder?
        ("Monte-Carlo Cropping Parameters", "mc_params", [
            lambda a, p, s, c: (("padded," if p else "") + str(s) + "," + str(c)) if a else "none",
            ("Apply Monte-Carlo Cropping (MCC)", "apply_mcc", True),
            ("Apply Padding", "padding", True),
            ("Crop Size", "crop_size", (896, 32, None, 32)),
            ("Crop Count", "crop_count", (2, 1, None, 1)),
        ]),
        ("Data Augmentation", "aug_strat", [
            lambda *args: ",".join(["hflip", "vflip", "rot90"][i] for i in range(len(args)) if args[i]),
            ("Random Horizontal Flipping", "hflip", True),
            ("Random Vertical Flipping", "vflip", True),
            ("Random 90° Rotation", "rot90", True),
        ]),
        ("Batch Size", "batch_size", (1, 1, None, 1)),
        ("Max. Number of Workers", "max_workers", (16, 1, 32, 1)),
    ]

class CoverModelParameterBlock(ArgumentBasedParameterBlock):
    title = "Cover Model"

    arguments = [
        ("Cover Head", "cover_head_type", ([
            ("Sigmoidal (with Background and Irrelevance)", "default"),
            ("Sigmoidal", "sigmoid"),
        ], "default")),
        ("Phenology Model", "pheno_model", ([
            ("Flowering/Senescence", "fs"),
            ("None", "none"),
        ], "fs")),
    ]
