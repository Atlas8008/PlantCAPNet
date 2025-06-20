import gradio as gr

from .layout import Layout
from .. import blocks


class TrainingLayout(Layout):
    def __init__(self, with_cover_training):
        super().__init__()

        self.with_cover_training = with_cover_training

    def layout(self, fn, tracker, root):
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Training Parameters", open=False):
                    blocks.MetaParameterBlock(
                        zeroshot=not self.with_cover_training,
                    )(tracker=tracker, root=root)
                with gr.Accordion("Slurm Usage", open=False):
                    # Slurm Usage
                    blocks.SlurmParameterBlock()(
                        tracker=tracker,
                        root=root,
                    )
                with gr.Accordion("Datasets", open=False):
                    # Slurm Usage
                    blocks.DatasetParameterBlock(                        zeroshot=not self.with_cover_training
                    )(
                        tracker=tracker,
                        root=root,
                    )
            with gr.Column():
                # Classification Pre-Training
                with gr.Accordion("Classification Pre-Training", open=False):
                    blocks.ClassificationTrainingParameterBlock(
                        tracker_prefix="c_",
                    )(tracker=tracker, root=root)
                    blocks.ClassificationModuleParameterBlock(
                        tracker_prefix="cm_"
                    )(tracker=tracker, root=root)
                with gr.Accordion("Classification/WSOL Network", open=False):
                    blocks.FeatureExtractorParameterBlock(kind="wsol", zeroshot=not self.with_cover_training, tracker_prefix="wf_"
                    )(tracker=tracker, root=root)

            with gr.Column():
                # Segmentation Pre-Training
                with gr.Accordion("Segmentation Pre-Training", open=False):
                    blocks.SegmentationTrainingParameterBlock(
                        is_zeroshot=not self.with_cover_training,
                        tracker_prefix="s_",
                    )(tracker=tracker, root=root)
                with gr.Accordion("Segmentation and Cover Network", open=False):
                    blocks.FeatureExtractorParameterBlock(kind="segmentation", zeroshot=not self.with_cover_training, tracker_prefix="sf_"
                    )(tracker=tracker, root=root)

            with gr.Column():
                # Cover Training
                if self.with_cover_training:
                        with gr.Accordion("Cover Training", open=False):
                            blocks.CoverTrainingParameterBlock(
                                tracker_prefix="cp_",
                            )(tracker=tracker, root=root)
                            blocks.CoverModelParameterBlock(
                                tracker_prefix="cpm_",
                            )(tracker=tracker, root=root)
                with gr.Accordion("Ensemble", open=False):
                    blocks.CoverEnsembleParameterBlock(
                        zeroshot=not self.with_cover_training,
                    )(tracker=tracker, root=root)


        btn = gr.Button()

        out = gr.Code()

        btn.click(
            fn,
            inputs=tracker.dump_list(),
            outputs=[out],
        )
