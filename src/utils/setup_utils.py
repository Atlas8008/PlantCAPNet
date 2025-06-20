from src import engines

model_kwargs_cover_trained = {
    "head": "cover_prediction",
    "mode": ("segmentation", "cover_prediction"),
}
model_kwargs_zeroshot = {
    "head": "zeroshot_cover_prediction",
    "mode": ("segmentation", "cover_prediction"),
    "interlaced_prediction": True,
    "submodel_kwargs": {
        "head": "segmentation",
        "use_deocclusion": False,
        "segmentation_kwargs": {
            "mode": "cam",
        },
        "deocclusion_kwargs": {
            "mode": "cam",
        }
    }
}

model_defaults = {
    "engine": "Standardized",
    "preprocessing": "torch",
    "input_width": 3200,
    "input_height": 1792,
}


class Setup:
    def __init__(self):
        self._cover_training_engine = None
        self._zeroshot_training_engine = None
        self._cover_engine_manager = None
        self._zeroshot_engine_manager = None

    def initialize(self, device, model_config, model_restrictions=None, cover_engine_manager=None, zeroshot_engine_manager=None):
        """
        Initialize the setup with the provided arguments and model configuration.

        :param device: The device to be used for model inference (e.g., 'cuda:0').
        :param model_config: Configuration for the models to be used.
        :param model_restrictions: Optional list of model names to restrict the setup to.
        :param cover_engine_manager: Optional pre-initialized cover engine manager.
        :param zeroshot_engine_manager: Optional pre-initialized zeroshot engine manager.
        """
        if model_restrictions is None:
            model_restrictions = []

        self._cover_training_engine = engines.training.TrainingEngine(cover_training=True)
        self._zeroshot_training_engine = engines.training.TrainingEngine(cover_training=False)

        cover_settings = model_config["cover-trained"].copy()
        zeroshot_settings = model_config["zero-shot"].copy()

        for k, c in cover_settings.items():
            cover_settings[k] = {**model_defaults, **c}
        for k, c in zeroshot_settings.items():
            zeroshot_settings[k] = {**model_defaults, **c}

        cover_engines = {
            k: engines.prediction.ENGINES[c["engine"]](
                c["model"],
                device,
                preprocessing=c["preprocessing"],
                input_size_hw=(c["input_height"], c["input_width"]),
                model_kwargs=model_kwargs_cover_trained,
            ) for k, c in cover_settings.items() if not model_restrictions or any(k in r for r in model_restrictions)
        }
        zeroshot_engines = {
            k: engines.prediction.ENGINES[c["engine"]](
                c["model"],
                device,
                preprocessing=c["preprocessing"],
                input_size_hw=(c["input_height"], c["input_width"]),
                model_kwargs=model_kwargs_zeroshot,
                with_phenology=False,
            ) for k, c in zeroshot_settings.items() if not model_restrictions or any(k in r for r in model_restrictions)
        }

        if cover_engine_manager is None:
            self._cover_engine_manager = engines.PredictionEngineManager(
                cover_engines,
                with_phenology=True,
                zeroshot=False,
            )
        else:
            self._cover_engine_manager = cover_engine_manager
            self._cover_engine_manager.engine_dict = cover_engines

        if zeroshot_engine_manager is None:
            self._zeroshot_engine_manager = engines.PredictionEngineManager(
                zeroshot_engines,
                with_phenology=False,
                zeroshot=True,
            )
        else:
            self._zeroshot_engine_manager = zeroshot_engine_manager
            self._zeroshot_engine_manager.engine_dict = zeroshot_engines

        # if cover_engine_manager is not None:
        #     m_new, m_old = self.cover_engine_manager, cover_engine_manager

        #     m_new.en

        #     m_new._single_prediction_layout = m_old._single_prediction_layout
        #     m_new._batch_prediction_layout = m_old._batch_prediction_layout
        #     m_new._single_pred_value_tracker = m_old._single_pred_value_tracker
        #     m_new._batch_pred_value_tracker = m_old._batch_pred_value_tracker

        # if zeroshot_engine_manager is not None:
        #     m_new, m_old = self.zeroshot_engine_manager, zeroshot_engine_manager

        #     m_new._single_prediction_layout = m_old._single_prediction_layout
        #     m_new._batch_prediction_layout = m_old._batch_prediction_layout
        #     m_new._single_pred_value_tracker = m_old._single_pred_value_tracker
        #     m_new._batch_pred_value_tracker = m_old._batch_pred_value_tracker

    @property
    def cover_training_engine(self):
        """
        Get the cover training engine.

        :return: The cover training engine.
        """
        return self._cover_training_engine

    @property
    def zeroshot_training_engine(self):
        """
        Get the zeroshot training engine.

        :return: The zeroshot training engine.
        """
        return self._zeroshot_training_engine

    @property
    def cover_engine_manager(self):
        """
        Get the cover engine manager.

        :return: The cover engine manager.
        """
        return self._cover_engine_manager

    @property
    def zeroshot_engine_manager(self):
        """
        Get the zeroshot engine manager.

        :return: The zeroshot engine manager.
        """
        return self._zeroshot_engine_manager


