from .parameter_block import ArgumentBasedParameterBlock


class DatasetParameterBlock(ArgumentBasedParameterBlock):
    title = "Dataset Parameters"

    def __init__(self, zeroshot=False, tracker_prefix=""):
        super().__init__(tracker_prefix)

        self.arguments = [
            ("Pre-Training Dataset", "pt_dataset_name", "example_ds", {"info": "Dataset to use for model pre-training. Custom datasets are also supported; in this case the name of the dataset folder (located under datasets/) has to be provided."}),
            ("Community Dataset", "community_dataset_name", "InsectArmageddon_test" if not zeroshot else "None", {"info": "Dataset to use for plant community training, evaluation or model building. Custom datasets are also supported; in this case the name of the dataset folder (located under datasets/) has to be provided."}),
        ]
