
import gradio as gr

from src.layouters import layouts
from src.utils import ComponentValueTracker


class EngineManager:
    def __init__(self, engine_dict) -> None:
        self.engine_dict = engine_dict


class PredictionEngineManager(EngineManager):
    def __init__(self, engine_dict, with_phenology, zeroshot) -> None:
        super().__init__(engine_dict)

        self.with_phenology = with_phenology
        self.zeroshot = zeroshot

        self._single_pred_value_tracker = ComponentValueTracker()
        self._batch_pred_value_tracker = ComponentValueTracker()

        self._single_prediction_layout = None
        self._batch_prediction_layout = None

    def single_prediction(self, engine_name, image, *params):
        if engine_name not in self.engine_dict:
            raise gr.Error(f"Invalid model selected, please add or select a valid model")

        params_dict = ComponentValueTracker.tracked_list_to_dict(
            self._single_pred_value_tracker,
            params,
        )

        return self.engine_dict[engine_name].single_prediction(image, **params_dict)

    def batch_prediction(self, engine_name, *params, progress=gr.Progress()):
        if engine_name not in self.engine_dict:
            raise gr.Error(f"Invalid model selected, please add or select a valid model")

        params_dict = ComponentValueTracker.tracked_list_to_dict(
            self._batch_pred_value_tracker,
            params,
        )

        return self.engine_dict[engine_name].batch_prediction(progress=progress, **params_dict)


    def single_prediction_layout(self, models, examples=False, *args, **kwargs):
        self._single_prediction_layout = layouts.SinglePredictionLayout(
            models=models,
            with_phenology=self.with_phenology,
            examples=examples,
        )
        self._single_prediction_layout(
            self.single_prediction,
            self._single_pred_value_tracker,
            *args,
            **kwargs,
        )

    def batch_prediction_layout(self, models, *args, **kwargs):
        self._batch_prediction_layout = layouts.BatchPredictionLayout(
            models=models,
            with_phenology=self.with_phenology,
            zeroshot=self.zeroshot,
        )
        self._batch_prediction_layout(
            self.batch_prediction,
            self._batch_pred_value_tracker,
            *args,
            **kwargs,
        )

