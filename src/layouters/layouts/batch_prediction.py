import os
import pandas as pd
import gradio as gr

from PIL import Image
from src.utils import utils

from .layout import Layout
from ..blocks import TimeSeriesPostProcessingParameterBlock, ZeroshotInferenceParameterBlock, CoverInferenceParameterBlock


def update_file(files, format, aggregate):
    """
    Update the file list and format based on the provided datetime format.
    Args:
        files (list): List of file paths.
        format (str): Datetime format string.
        aggregate (str): Aggregation level for datetime.
    Returns:
        tuple: Updated file list and a DataFrame with datetime information.
    """

    if files is None:
        return None, None

    files = sorted(files, key=lambda p: os.path.basename(p))

    basenames = [os.path.basename(f) for f in files]

    if format == "None": # If timedate format is not provided, use only filenames
        output_table = pd.DataFrame({"filenames": basenames})
    else:
        try:
            output_table = utils.filenames_to_datetime_table(basenames, format)

            if aggregate != "None":
                times = ["year", "month", "day", "hour", "minute"]
                times = times[:times.index(aggregate.lower()) + 1]

                output_table = output_table.drop("filenames", axis=1).groupby(times).mean().index.to_frame()
                print(output_table)
        except Exception as e:
            output_table = pd.DataFrame({"Invalid data format": ["Datetime could not be correctly extracted from the filenames with the provided datetime format."]})
            print(e)

    print(files)
    return files, output_table


class BatchPredictionLayout(Layout):
    def __init__(self, models, with_phenology, zeroshot=False) -> None:
        """
        Initializes the batch prediction layout.

        Args:
            models (list): A list of models to be used for batch prediction.
            with_phenology (bool): Indicates whether phenology prediction is included.
            zeroshot (bool, optional): Specifies if zero-shot learning is enabled. Defaults to False.
        """
        self.models = models
        self.with_phenology = with_phenology
        self.zeroshot = zeroshot
        self.model_dropdown = None

    def layout(self, predict_fn, tracker, root):
        self.model_dropdown = model = gr.Dropdown(self.models, value=self.models[0], label="Model")
        with gr.Group():
            with gr.Row():
                gr.Markdown("## &nbsp;Input")
            with gr.Row():
                # Image filetypes readable by PIL
                image_filetypes = [
                    ".jpg", ".jpeg", ".png", ".bmp",
                    ".gif", ".tiff", ".webp", ".tif"
                ]

                file = gr.File(
                    file_count="multiple", file_types=image_filetypes, label="Input images", height=300)

                tracker["files"] = file

                projected_output_format = gr.DataFrame(col_count=0, label="Projected output format", max_height=300)

        with gr.Group():
            with gr.Row():
                gr.Markdown("## &nbsp;Parameters")

            with gr.Accordion("Temporal Processing Parameters", open=False):
                TimeSeriesPostProcessingParameterBlock()(tracker, root=root)

                with gr.Row():
                    with gr.Column():
                        tracker["timedate_format"] = gr.Dropdown(
                            choices=[
                                "None",
                                "%Y-%m-%dT%H:%M:%S",
                                "%Y-%m-%dT%H%M%S",
                                "%Y-%m-%d-%H:%M:%S",
                                "%Y-%m-%d-%H-%M-%S",
                                "%Y-%m-%d-%H%M%S",
                                "%Y%m%d-%H%M%S",
                                "%Y%m%d%H%M%S",
                                "%Y-%m-%d",
                            ],
                            allow_custom_value=True,
                            value="None",
                            label="Datetime format in filenames (see also https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)"
                        )

                    with gr.Column():
                        tracker["aggregate"] = gr.Dropdown(
                            choices=[
                                "None",
                                "Minute",
                                "Hour",
                                "Day",
                                "Month",
                                "Year",
                            ],
                            value="None",
                            label="The predictions will be averaged over per this unit of time."
                        )

            if self.zeroshot:
                with gr.Accordion("Zero-Shot Inference Parameters"):
                    ZeroshotInferenceParameterBlock()(
                        tracker=tracker, root=root
                    )
            else:
                with gr.Accordion("Inference Parameters"):
                    CoverInferenceParameterBlock()(
                        tracker=tracker, root=root
                    )

        file_format_check = {
            "fn": update_file,
            "inputs": [file, tracker["timedate_format"], tracker["aggregate"]],
            "outputs": [file, projected_output_format],
        }

        file.upload(**file_format_check)
        tracker["timedate_format"].change(**file_format_check)
        tracker["aggregate"].change(**file_format_check)

        with gr.Row():
            btn = gr.Button(
                "Predict",
            )

        dataframes = []
        download_btns = []
        plots = []

        with gr.Group():
            with gr.Row():
                gr.Markdown("# &nbsp;Outputs")
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## &nbsp;Plant Cover")
                    with gr.Column(scale=1):
                        cover_download = gr.DownloadButton()
                with gr.Row():
                    with gr.Column(scale=1):
                        cover_frame = gr.DataFrame(col_count=1)
                    with gr.Column(scale=1):
                        cover_plot = gr.LinePlot(height=600)

                dataframes.append(cover_frame)
                download_btns.append(cover_download)
                plots.append(cover_plot)

            if self.with_phenology:
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## &nbsp;Flowering Phenology")
                        with gr.Column(scale=1):
                            flow_download = gr.DownloadButton()
                    with gr.Row():
                        with gr.Column():
                            flowering_frame = gr.DataFrame(col_count=1)
                        with gr.Column():
                            flowering_plot = gr.LinePlot(height=600)

                dataframes.append(flowering_frame)
                download_btns.append(flow_download)
                plots.append(flowering_plot)

                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## &nbsp;Senescence Phenology")
                        with gr.Column(scale=1):
                            sen_download = gr.DownloadButton()
                    with gr.Row():
                        with gr.Column():
                            senescence_frame = gr.DataFrame(col_count=1)
                        with gr.Column():
                            senescence_plot = gr.LinePlot(height=600)

                dataframes.append(senescence_frame)
                download_btns.append(sen_download)
                plots.append(senescence_plot)

        btn.click(
            fn=predict_fn,
            inputs=[model] + tracker.dump_list(),
            outputs=dataframes + download_btns + plots,
        )
