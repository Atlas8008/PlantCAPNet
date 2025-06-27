import os
import gc
import torch
import ctypes
import tempfile
import numpy as np
import gradio as gr
import pandas as pd

from torch import nn
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from src.utils import filenames_to_datetime_table

import plantcapnet_compute as cap
from plantcapnet_compute.utils import augmentations as augs


def trim_memory(): # Solves pytorch memory leak, see https://github.com/pytorch/pytorch/issues/68114
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

OUTPUT_SCALE = 1

LOADED_MODELS = {}


def resize(img, resolution_hw):
    """
    Resizes an image to the specified resolution using bicubic interpolation.

    Args:
        img (PIL.Image.Image): The image to resize.
        resolution_hw (tuple): Target resolution as (height, width).

    Returns:
        PIL.Image.Image: Resized image.
    """
    return transforms.Resize(
        resolution_hw,
        interpolation=transforms.InterpolationMode.BICUBIC
    )(img)


def get_transforms(preprocessing):
    """
    Creates a transformation pipeline for preprocessing images.

    Args:
        preprocessing (str): Preprocessing mode.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        augs.Preprocess(mode=preprocessing),
    ])


def plot_bar(df: pd.DataFrame, y_label, title):
    """
    Plots a horizontal bar chart for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: The generated bar chart figure.
    """
    plt.style.use("dark_background")

    print(df)

    axes = df.plot.barh("Species", y_label, title=title)
    axes.set_xlim([0, 100])
    fig = axes.get_figure()
    fig.tight_layout()

    return fig


class ModelManager:
    """
    Manages the loading and unloading of models, ensuring memory is freed after use.

    Attributes:
        model_path (str): Path to the model file.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').
        set_device_dynamically (bool): Whether to dynamically set the device.
        model (torch.nn.Module): The loaded model.
    """
    def __init__(self, model_path, device, set_device_dynamically=None):
        """
        Initializes the ModelManager.

        Args:
            model_path (str): Path to the model file.
            device (str): Device to load the model on.
            set_device_dynamically (bool): Whether to dynamically set the device. If None, will determine based on if the model has a dynamic_device attribute.
        """
        self.model_path = model_path
        self.device = device
        self.set_device_dynamically = set_device_dynamically
        self.model = None

    def __enter__(self):
        """
        Loads the model when entering the context.

        Returns:
            ModelManager: The current instance with the loaded model.
        """
        global LOADED_MODELS


        if self.model_path not in LOADED_MODELS:
            LOADED_MODELS = {}
            print("Loading model from", self.model_path)
            self.model = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=False,
            )

            LOADED_MODELS[self.model_path] = self.model
        else:
            print("Reusing model for", self.model_path)
            self.model = LOADED_MODELS[self.model_path]

        set_device_dynamically = self.set_device_dynamically

        if set_device_dynamically is None:
            # If the model is an ensemble model, set device dynamically
            set_device_dynamically = hasattr(self.model, "dynamic_device")

        device = "cpu" if set_device_dynamically else self.device

        if set_device_dynamically:
            self.model.dynamic_device = self.device

        self.model.to(device)

        return self

    def __exit__(self, exc, value, traceback):
        """
        Frees memory and unloads the model when exiting the context.

        Args:
            exc (Exception): Exception raised, if any.
            value (Any): Exception value.
            traceback (traceback): Traceback object.
        """
        if exc:
            print(traceback)

        #del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        trim_memory()
        print("Memory freed")

class PredictionEngine:
    """
    Handles predictions for images and batches, including preprocessing and postprocessing.

    Attributes:
        model (str): Path to the model.
        device (str): Device to run predictions on.
        preprocessing (str): Preprocessing mode.
        input_size_hw (tuple): Input size as (height, width).
        model_kwargs (dict): Additional arguments for the model.
        with_phenology (bool): Whether to include phenology predictions.
    """
    def __init__(self, model, device, preprocessing, input_size_hw, model_kwargs, with_phenology=True) -> None:
        """
        Initializes the PredictionEngine.

        Args:
            model (str): Path to the model.
            device (str): Device to run predictions on.
            preprocessing (str): Preprocessing mode.
            input_size_hw (tuple): Input size as (height, width).
            model_kwargs (dict): Additional arguments for the model.
            with_phenology (bool): Whether to include phenology predictions.
        """
        self._model_str = model
        self.device = device
        self.preprocessing = preprocessing
        self.input_size_hw = input_size_hw
        self.model_kwargs = model_kwargs

        self.transforms = get_transforms(preprocessing=preprocessing)
        self.with_phenology = with_phenology

        self.temp_files = {}

    def get_tspp_model(self, type, kernel_size, kernel_base, kernel_sigma):
        """
        Creates a time-series postprocessing model.

        Args:
            type (str): Type of kernel.
            kernel_size (int): Size of the kernel.
            kernel_base (float): Base value for the kernel.
            kernel_sigma (float): Sigma value for Gaussian kernel.

        Returns:
            MovingAverageTimeSeriesModel: The created time-series model.
        """
        mode = {
            "Moving Average - Constant Kernel": "constant",
            "Moving Average - Exponential Kernel": "exponential",
            "Moving Average - Linear Kernel": "linear",
            "Moving Average - Gaussian Kernel": "gaussian",
        }[type]

        if mode == "constant":
            kernel_base = 1.0
            mode = "exponential"

        model = cap.models.time_series.MovingAverageTimeSeriesModel(
            n=kernel_size,
            base=kernel_base,
            alignment="center",
            sigma=kernel_sigma,
            mode=mode,
        )

        return model

    @torch.no_grad()
    def predict_image(self, model, img, with_segmentation, species_labels, input_size=None, filter_threshold=None):
        """
        Predicts the output for a single image.

        Args:
            model (torch.nn.Module): The model to use for prediction.
            img (PIL.Image.Image or np.ndarray): The input image.
            with_segmentation (bool): Whether to include segmentation in the output.
            species_labels (list): List of species labels.
            input_size (tuple, optional): Input size as (height, width). Defaults to None.
            filter_threshold (float, optional): Threshold for filtering predictions. Defaults to None.

        Returns:
            dict: Dictionary containing prediction results.
        """
        model.eval()
        output_dict = {}

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if input_size is None:
            input_size = self.input_size_hw

        orig_img = resize(img, input_size)

        print(self.preprocessing)
        img = self.transforms(orig_img)[None].to(self.device)

        with torch.amp.autocast(self.device):
            outputs = model(img, **self.model_kwargs)

        if isinstance(outputs, tuple):
            outputs_cover = outputs[0]
        else:
            outputs_cover = outputs
        cover_values = outputs_cover["cover_prediction"].detach().cpu().numpy()[0].astype("float32") * 100

        if self.with_phenology:
            outputs_phenology = outputs[1]
            phenology_values = outputs_phenology["phenology_prediction"]
            flowering_values = phenology_values[:, :, 0].detach().cpu().numpy()[0].astype("float32") * 100
            senescence_values = phenology_values[:, :, 1].detach().cpu().numpy()[0].astype("float32") * 100

        if filter_threshold is not None:
            gt_threshold = cover_values > filter_threshold
            species_labels = [
                s for s, is_gt_thresh in zip(species_labels, gt_threshold) if is_gt_thresh
            ]

            cover_values = cover_values[gt_threshold]
            if self.with_phenology:
                flowering_values = flowering_values[gt_threshold]
                senescence_values = senescence_values[gt_threshold]
        else:
            gt_threshold = None

        output_dict["df_cover"] = pd.DataFrame({
            "Species": species_labels,
            "Cover (%)": cover_values,
        })
        if self.with_phenology:
            output_dict["df_flowering"] = pd.DataFrame({
                "Species": species_labels,
                "Flowering Intensity (%)": flowering_values,
            })
            output_dict["df_senescence"] = pd.DataFrame({
                "Species": species_labels,
                "Senescence Intensity (%)": senescence_values,
            })

        if with_segmentation:
            segmentation = outputs_cover["segmentation"]
            if self.with_phenology:
                segmentation_phenology = outputs_phenology["segmentation"]
                segmentation_flowering = segmentation_phenology[:, :, 0]
                segmentation_senescence = segmentation_phenology[:, :, 1]

            if gt_threshold is not None:
                # Include background and irrelevance, if they exist
                print(segmentation.shape[1], gt_threshold.shape[0])
                if segmentation.shape[1] > gt_threshold.shape[0]:
                    gt_threshold_seg = np.concatenate([
                        gt_threshold,
                        [True] * (segmentation.shape[1] - gt_threshold.shape[0])],
                        axis=0,
                    )
                else:
                    gt_threshold_seg = gt_threshold

                segmentation = segmentation[:, gt_threshold_seg]
                if self.with_phenology:
                    segmentation_flowering = segmentation_flowering[:, gt_threshold_seg[:segmentation_flowering.shape[1]]]
                    segmentation_senescence = segmentation_senescence[:, gt_threshold_seg[:segmentation_senescence.shape[1]]]

            segmentation = nn.functional.interpolate(
                segmentation,
                scale_factor=OUTPUT_SCALE,
                mode="bilinear",
            )
            if self.with_phenology:
                segmentation_flowering = nn.functional.interpolate(
                    segmentation_flowering,
                    scale_factor=OUTPUT_SCALE,
                    mode="bilinear",
                )
                segmentation_senescence = nn.functional.interpolate(
                    segmentation_senescence,
                    scale_factor=OUTPUT_SCALE,
                    mode="bilinear",
                )

            orig_img = transforms.Resize(
                segmentation.shape[-2:],
                interpolation=transforms.InterpolationMode.BICUBIC
            )(orig_img)

            output_dict["segmentation"] = segmentation.detach().cpu().numpy()[0].copy()
            if self.with_phenology:
                output_dict["segmentation_flowering"] = segmentation_flowering.detach().cpu().numpy()[0].copy()
                output_dict["segmentation_senescence"] = segmentation_senescence.detach().cpu().numpy()[0].copy()


        output_dict["orig_img"] = orig_img

        if gt_threshold is not None:
            return output_dict, gt_threshold.tolist()
        return output_dict

    def single_prediction(self, img, **params):
        """
        Performs a single prediction on an image.

        Args:
            img (PIL.Image.Image): The input image.
            **params: Additional parameters for prediction.

        Returns:
            tuple: Prediction results.
        """
        if params["use_custom_img_size"]:
            image_size = [int(v) for v in params["image_size"].split(" ")]
        else:
            image_size = None

        try:
            with ModelManager(self._model_str, self.device) as mm:
                pred, filter_mask = self.predict_image(
                    mm.model,
                    img,
                    with_segmentation=True,
                    species_labels=self.species_labels,
                    input_size=image_size,
                    filter_threshold=0.1,
                )
        except RuntimeError as e:
            gr.Warning(str(e), title="Error")
            print(e)

            if self.with_phenology:
                return img, *([None] * 6)
            else:
                return img, *([None] * 2)

        seg_labels = self.all_labels

        if filter_mask:
            if len(self.all_labels) > len(filter_mask):
                filter_mask += [True] * (len(self.all_labels) - len(filter_mask))

            seg_labels = [lbl for lbl, mask in zip(seg_labels, filter_mask) if mask]

        img = pred["orig_img"]
        cover_seg = [(subseg, lbl) for subseg, lbl in zip(pred["segmentation"], seg_labels)]
        outputs = [img, (img, cover_seg), pred["df_cover"]]

        if self.with_phenology:
            flow_seg = [(subseg, lbl) for subseg, lbl in zip(pred["segmentation_flowering"], seg_labels)]
            sen_seg = [(subseg, lbl) for subseg, lbl in zip(pred["segmentation_senescence"], seg_labels)]

            outputs.extend([
                (img, flow_seg),
                pred["df_flowering"],
                (img, sen_seg),
                pred["df_senescence"],
            ])

        outputs = tuple(outputs)

        return *outputs,

    @torch.no_grad()
    def batch_prediction(self, progress=gr.Progress(), **params):
        """
        Performs batch predictions on a list of images.

        Args:
            progress (gr.Progress): Progress bar for tracking.
            **params: Additional parameters for batch prediction.

        Returns:
            tuple: Batch prediction results.
        """
        image_files = params["files"]

        if params["use_custom_img_size"]:
            image_size = [int(v) for v in params["image_size"].split(" ")]
        else:
            image_size = None

        if not isinstance(image_files, list):
            return None

        if self.with_phenology:
            keys = ["cover", "flowering", "senescence"]
        else:
            keys = ["cover"]

        dfs = {
            k: [] for k in keys
        }
        try:
            with ModelManager(self._model_str, self.device) as mm:
                for image_file in progress.tqdm(image_files):
                    img = Image.open(image_file).convert("RGB")

                    pred = self.predict_image(
                        mm.model,
                        img,
                        with_segmentation=False,
                        species_labels=self.species_labels,
                        input_size=image_size,
                    )

                    for k in keys:
                        pred["df_" + k] = pred["df_" + k].set_index("Species").T
                        dfs[k].append(pred["df_" + k])

        except RuntimeError as e:
            gr.Warning(str(e), title="Error")
            print(e)

            return *([None] * len(keys) * 3),

        for k in keys:
            dfs[k] = pd.concat(dfs[k], axis="rows", ignore_index=True)

        filenames = [os.path.basename(image_file) for image_file in image_files]

        if params["timedate_format"] != "None":
            datetimes = filenames_to_datetime_table(filenames, format=params["timedate_format"])
            for df in dfs.values():
                df.index = pd.MultiIndex.from_frame(datetimes)
        else:
            datetimes = None
            for df in dfs.values():
                df.index = filenames


        # Average over unit here
        if params["aggregate"] != "None" and datetimes is not None:
            times = ["year", "month", "day", "hour", "minute"]
            times = times[:times.index(params["aggregate"].lower()) + 1]

            for k in keys:
                dfs[k] = dfs[k].droplevel("filenames").groupby(times).mean()

        # Employ time-series postprocessing model
        if params["tspp_type"] != "None" and datetimes is not None:
            tspp_model = self.get_tspp_model(
                type=params["tspp_type"],
                kernel_size=params["tspp_kernel_size"],
                kernel_base=params["tspp_kernel_base"],
                kernel_sigma=params["tspp_kernel_sigma"],
            )

            for df in dfs.values():
                ts_data = tspp_model(df.to_numpy())

                df.loc[:, :] = ts_data


            print("Applied time series model")


        for k in keys:
            self.temp_files[k] = tempfile.NamedTemporaryFile(
                prefix=k, suffix=".csv")
            dfs[k].to_csv(self.temp_files[k].name)

        plots = []

        for name, df in dfs.items():
            name = name.capitalize()

            df_display: pd.DataFrame = df.copy()

            if datetimes is not None:
                df_display = df_display.stack()
                df_display.index.set_names(level=-1, names="Species")
                df_display = df_display.reset_index(level="Species")

                datetime_indices = df_display.index.to_frame().drop("filenames", axis=1, errors="ignore")
                if "year" not in datetime_indices.columns:
                    datetime_indices["year"] = [2024] * len(datetime_indices)
                if "month" not in datetime_indices.columns:
                    datetime_indices["month"] = [1] * len(datetime_indices)
                if "day" not in datetime_indices.columns:
                    datetime_indices["day"] = [1] * len(datetime_indices)
                df_display["Datetime"] = pd.to_datetime(datetime_indices)
                df_display.columns = ["Species", name, "Datetime"]
                print(df_display)
                plot = gr.LinePlot(df_display.reset_index(drop=True), x="Datetime", y=name, color="Species")
            else:
                print(df_display)
                df_display = df_display.reset_index(drop=True)
                df_display = df_display.stack()
                df_display.index.set_names(level=-1, names="Species")
                df_display = df_display.reset_index(level="Species")

                df_display = df_display.reset_index()
                df_display.columns = ["Index", "Species", name]

                print(df_display)
                plot = gr.LinePlot(df_display, x="Index", y=name, color="Species")
            plots.append(plot)

        dfs = tuple(df.reset_index() for df in dfs.values())
        temp_file_names = tuple(f.name for f in self.temp_files.values())

        return *dfs, *temp_file_names, *plots


class StandardizedDatasetPredictionEngine(PredictionEngine):
    """
    Prediction engine for standardized datasets.
    """
    def __init__(self, *args, class_labels_path=None, **kwargs) -> None:
        """
        Initializes the GCEF23PredictionEngine.
        """
        super().__init__(*args, **kwargs)

        if class_labels_path is None:
            model_path, model_name = os.path.split(self._model_str)
            model_name = os.path.splitext(model_name)[0]

            class_labels_path = os.path.join(
                model_path,
                model_name + "_class-labels.txt"
            )

        with open(class_labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]

        labels_ext = labels + ["Background"]

        self.species_labels = labels
        self.all_labels = labels_ext



ENGINES = {
    "Standardized": StandardizedDatasetPredictionEngine,
}