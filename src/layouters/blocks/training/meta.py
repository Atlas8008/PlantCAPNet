from .parameter_block import ArgumentBasedParameterBlock


class MetaParameterBlock(ArgumentBasedParameterBlock):
    title = "Training Parameters"

    def __init__(self, zeroshot=False, tracker_prefix=""):
        super().__init__(tracker_prefix)
        self.arguments = [
            ("Training Name", "name", "model"),
            ("Run Evaluation", "run_evaluation", not zeroshot, {"info": "If set, the model is evaluated after training. If not set, the model is only trained."}),
        ]