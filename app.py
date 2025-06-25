import sys

sys.path.extend([
    "plantcapnet_compute/",
    "plantcapnet_compute/networks",
])

import os
import argparse
import gradio as gr
import yaml

from src import engines
from src.utils import Setup

def update_models(*dropdowns):
    with open("config/models.yaml", "r") as f:
        model_config = yaml.safe_load(f.read())

    setup.initialize(
        device=args.device,
        model_config=model_config,
        model_restrictions=model_restrictions,
        cover_engine_manager=setup.cover_engine_manager,
        zeroshot_engine_manager=setup.zeroshot_engine_manager,
    )
    models_cover = list(setup.cover_engine_manager.engine_dict.keys())
    models_zeroshot = list(setup.zeroshot_engine_manager.engine_dict.keys())

    if len(models_cover) > 0:
        selected_cover_model = models_cover[0]
    else:
        selected_cover_model = None

    if len(models_zeroshot) > 0:
        selected_zeroshot_model = models_zeroshot[0]
    else:
        selected_zeroshot_model = None

    print("Model configuration updated.")
    return [
        gr.Dropdown(models_cover, value=selected_cover_model)
    ] * (len(dropdowns) // 2) + [
        gr.Dropdown(models_zeroshot, value=selected_zeroshot_model)
    ] * (len(dropdowns) // 2)

MODES = ("covertrained", "zeroshot", "training")

os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.path.dirname(__file__), ".gradio_cache")

static_paths = ["static/"]

gr.set_static_paths(paths=static_paths)


parser = argparse.ArgumentParser(description="This web application is for model inference for plant cover and phenology prediction.")

parser.add_argument("--setups", nargs="*", type=str, default=None, help="The setups to restrict the app to, as defined in the models.json.")
parser.add_argument("--modes", nargs="*", type=str, default=None, choices=MODES, help="The modes to restrict the app to. By default, all modes are available.")
parser.add_argument("--examples", action="store_true", default=False, help="Flag; if set, examples are shown in the app.")
parser.add_argument("--device", type=str, default="cuda:0", help="The device to use for inference. Default is 'cuda:0'.")

args = parser.parse_args()

if args.modes is None:
    args.modes = MODES

with open("config/styles.css", "r") as f:
    css = f.read()
with open("config/scripts.js", "r") as f:
    js = f.read()
with open("config/models.yaml", "r") as f:
    model_config = yaml.safe_load(f.read())

model_restrictions = args.setups

setup = Setup()
setup.initialize(
    device=args.device,
    model_config=model_config,
    model_restrictions=model_restrictions,
)

with gr.Blocks(
        css=css,
        js=js,
        fill_height=True,
        delete_cache=(86400, 86400),
        theme=gr.themes.Origin(),
    ) as demo:
    with gr.Row():
        with gr.Column(scale=8):
            pass
        with gr.Column(scale=1):
            model_scan = gr.Button(
                "Scan for Model Changes",
                variant="secondary",
                elem_classes="model-scan-button",
                elem_id="model-scan-button",
            )
    if "covertrained" in args.modes:
        with gr.Tab("Cover-Trained Cover Prediction", elem_classes="tab"):
            with gr.Tab("Single", elem_classes="tab"):
                gr.Markdown("# &nbsp;Plant Community Analysis")
                setup.cover_engine_manager.single_prediction_layout(
                    models=list(setup.cover_engine_manager.engine_dict.keys()),
                    root=demo,
                    examples=args.examples,
                )

            with gr.Tab("Batched", elem_classes="tab"):
                gr.Markdown("# &nbsp;Batched Plant Community Analysis")
                setup.cover_engine_manager.batch_prediction_layout(
                    models=list(setup.cover_engine_manager.engine_dict.keys()),
                    root=demo,
                )
            if "training" in args.modes:
                with gr.Tab("Training", elem_classes="tab"):
                    gr.Markdown("# &nbsp;Cover-Trained Cover Prediction Training")
                    setup.cover_training_engine.training_layout(root=demo)

    if "zeroshot" in args.modes:
        with gr.Tab("Zero-Shot Cover Prediction", elem_classes="tab"):
            with gr.Tab("Single", elem_classes="tab"):
                gr.Markdown("# &nbsp;Single Zero-Shot Cover Prediction")
                setup.zeroshot_engine_manager.single_prediction_layout(
                    models=list(setup.zeroshot_engine_manager.engine_dict.keys()),
                    root=demo,
                    examples=args.examples,
                )
            with gr.Tab("Batched", elem_classes="tab"):
                gr.Markdown("# &nbsp;Batched Zero-Shot Cover Prediction")
                setup.zeroshot_engine_manager.batch_prediction_layout(
                    models=list(setup.zeroshot_engine_manager.engine_dict.keys()),
                    root=demo
                )
            if "training" in args.modes:
                with gr.Tab("Training", elem_classes="tab"):
                    gr.Markdown("# &nbsp;Zero-Shot Cover Prediction Training")
                    setup.zeroshot_training_engine.training_layout(root=demo)

    # Update of models in the dropdowns by re-checking the model configuration file
    model_scan.click(
        update_models,
        inputs=[
            setup.cover_engine_manager._single_prediction_layout.model_dropdown,
            setup.cover_engine_manager._batch_prediction_layout.model_dropdown,
            setup.zeroshot_engine_manager._single_prediction_layout.model_dropdown,
            setup.zeroshot_engine_manager._batch_prediction_layout.model_dropdown
        ],
        outputs=[
            setup.cover_engine_manager._single_prediction_layout.model_dropdown,
            setup.cover_engine_manager._batch_prediction_layout.model_dropdown,
            setup.zeroshot_engine_manager._single_prediction_layout.model_dropdown,
            setup.zeroshot_engine_manager._batch_prediction_layout.model_dropdown
        ],
    )


demo.launch(share=True)