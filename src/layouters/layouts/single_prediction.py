from .layout import Layout

import gradio as gr

import src.custom_components as cc

from ..blocks import CoverInferenceParameterBlock


class SinglePredictionLayout(Layout):
    def __init__(self, models, with_phenology, examples=True) -> None:
        """
        Initializes the single prediction layout.
        Args:
            models (list): A list of models to be used for single prediction.
            with_phenology (bool): Indicates whether phenology prediction is included.
            examples (bool, optional): Specifies if example images are included. Defaults to True.
        """
        self.models = models
        self.with_phenology = with_phenology
        self.examples = examples
        self.model_dropdown = None

    def layout(self, process_image, tracker, root):
        outputs = []

        self.model_dropdown = model = gr.Dropdown(self.models, label="Model", value=self.models[0] if self.models else None)

        with gr.Group():
            with gr.Accordion("Inference Parameters"):
                CoverInferenceParameterBlock()(
                    tracker=tracker, root=root,
                )

        with gr.Group():
            with gr.Row():
                gr.Markdown("## &nbsp;Input")
            with gr.Row(variant="panel") as input_row:
                with gr.Column(scale=3):
                    input = cc.Image(format="png", height=400)

        outputs.append(input)
        width = 300

        predict_btn = gr.Button("Predict")

        with gr.Group():
            with gr.Row():
                gr.Markdown("## &nbsp;Plant Cover")
            with gr.Row(variant="panel", equal_height=True):
                output_image = gr.AnnotatedImage(
                    show_label=False,
                    scale=3,
                )
                output_cover_perc = gr.BarPlot(
                    x="Species",
                    y="Cover (%)",
                    y_lim=[0, 100],
                    title="Species-wise Plant Cover",
                    vertical=False,
                    width=width,
                    container=False,
                    x_label_angle=45,
                    height=600,
                    scale=2,
                    sort="-y",
                )

            outputs.append(output_image)
            outputs.append(output_cover_perc)

        if self.with_phenology:
            with gr.Group():
                with gr.Row():
                        gr.Markdown("## &nbsp;Flowering Phenology")
                with gr.Row(variant="panel", equal_height=True):
                    output_image_flowering = gr.AnnotatedImage(
                        show_label=False,
                        scale=3,
                    )
                    output_flowering_perc = gr.BarPlot(
                        x="Species",
                        y="Flowering Intensity (%)",
                        y_lim=[0, 100],
                        title="Species-wise Flowering",
                        vertical=False,
                        width=width,
                        container=False,
                        x_label_angle=45,
                        scale=2,
                        height=600,
                        sort="-y",
                    )
                outputs.append(output_image_flowering)
                outputs.append(output_flowering_perc)

            with gr.Group():
                with gr.Row():
                    gr.Markdown("## &nbsp;Senescence Phenology")
                with gr.Row(variant="panel", equal_height=True):
                    output_image_senescence = gr.AnnotatedImage(
                        show_label=False,
                        scale=3,
                    )
                    output_senescence_perc = gr.BarPlot(
                        x="Species",
                        y="Senescence Intensity (%)",
                        y_lim=[0, 100],
                        title="Species-wise Senescence",
                        vertical=False,
                        width=width,
                        container=False,
                        x_label_angle=45,
                        scale=2,
                        height=600,
                        sort="-y",
                    )

                outputs.append(output_image_senescence)
                outputs.append(output_senescence_perc)

        input.upload(
            fn=process_image,
            inputs=[model, input] + tracker.dump_list(),
            outputs=outputs,

        )
        predict_btn.click(
            fn=process_image,
            inputs=[model, input] + tracker.dump_list(),
            outputs=outputs,
        )

        if self.examples:
            with input_row:
                with gr.Column(scale=2):
                    examples = gr.Examples(
                        inputs=[model, input] + tracker.dump_list(),
                        examples_per_page=24,
                        examples="example_images",
                        cache_examples="lazy",
                        outputs=outputs,
                        fn=process_image,
                        run_on_click=True,
                    )

def clear_outputs(outputs):
    def clear_fn():
        return tuple([None] * len(outputs))

    return clear_fn