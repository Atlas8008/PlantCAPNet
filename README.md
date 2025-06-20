# PlantCAPNet - **Plant** **C**ommunity **A**bundance and **P**henology **Net**work

PlantCAPNet is a web application for automatically determining the plant cover (plant abundances) and phenology of plant communities from top-down images. It comprises a web interface for configuring training and inference, and a backend used for the computations.

## Setup

The system was developed and tested on Ubuntu 22.04 LTS and 24.04 LTS. While it also might work under Windows, this is currently untested.

For easy setup, we recommend using [anaconda](https://www.anaconda.com/download) to manage python environments and packages. When using a working installation, run the installation script:

```
source run_install.sh
```
This will create a new conda environment and install all required dependencies in it. Note that the script should be run with `source` instead of `bash` or `sh`. Running with either of these might cause issues during the installation.

## Launching the Web-Interface

After installing the requirements and activating the corresponding conda environment, simply run the interface with

```
python app.py
```
or
```
gradio app.py
```

Also note the optional arguments of the application (`python app.py --help`).

By default, after running, the application should be accessible in your web browser under http://127.0.0.1:7860.

## The Web-Interface

### Single Prediction Interface

The Single Prediction Interface is for detailed inspection and validation of model predictions on one image at a time. It allows users to visually assess performance by displaying species-specific heatmaps (showing pixel-wise likelihood for plants, flowers, and senescent areas) and summary bar plots for cover and phenology percentages. This aids users in confirming prediction correctness.

### Batched Prediction Interface

The Batched Prediction Interface processes an entire series of images simultaneously, ideal for large datasets or time-series analysis. It can interpret file names as dates for chronological processing, enabling temporal aggregation (averaging predictions over time units like days or months for robustness) and temporal smoothing (using neighboring data to refine predictions). Results are presented in tables and line plots for cover and phenology, and can be downloaded as CSV files for further ecological analysis.

### The Training Interface

The Training Interface offers comprehensive configuration for the entire model training pipeline. Key areas include:

**Pre-training Configuration**: Allows setting parameters (learning rate, dataset, loss, architecture) for both classification and segmentation pre-training, including specific augmentations like Inverted Cutout.

**Plant Cover Training Configuration**: Enables defining task-specific hyperparameters, target datasets, and parameters for cover/phenology calculations.

**Ensemble Methods**: Supports configuring and building model ensembles (e.g., using varied architectures, different epochs, or repeated runs) to improve generalization.

**Execution Environment**: Provides options for local execution or distributed training on a SLURM cluster, with streamlined cluster submission settings.

After configuration, the interface generates the corresponding code, which can be copied or downloaded as a script for execution via the backend framework.


## The Backend (PlantCAPNet-Compute)

The backend is used to perform all the computations like training and inference, and can also be used independently for training on a different platform than the web interface, in case the resources of the web host are insufficient.

## Training Workflow

To train a custom model, you might want to prepare an own dataset. Any custom datasets used for training, should be put into or linked to (`ln -s`) the `datasets/`folder. For zero-shot cover prediction or pre-training, we recommend using the GBIF-Downloader (link), which already produces the correct format for being read into the backend system. If you have your own data for cover and phenology training, you have to bring it into the right format, which is specified [here](plantcapnet_compute/datasets/dataset_structure.md)

An example workflow using PlantCAPNet could look like the following. It should be noted that in the following example, the local backend is used for training. However, the backend can also be deployed independently on a different host for better hardware or higher compute capacity, like a cluster node, in which case the data has to be copied to and from that location.

### Pre-Training Dataset

To have a pre-training dataset, download one with the [GBIF downloader](gbif-image-downloader). Please set up the downloader beforehand as described in the [tool's respective README](gbif-image-downloader/README.md).

```bash
# Change to downloader directory
cd gbif-image-downloader/

# Download the dataset, download 10 images per species
python download_dataset.py "dataset_lists/example_ds.txt" 10

# After the tool is finished, move to the dataset folder...
cd "datasets/example_ds"

# And create a training and validation split of
# 8 and 2 images per class, respectively
python create_train_val_split.py 8
# Now we have a data.json containing the data for for training and validation splits

# Now, move the dataset to the dataset path of the backend
cd ../../../
mv -v gbif-image-downloader/datasets/example_ds plantcapnet_compute/datasets
```

Now the pre-training dataset is prepared and can be used for training in the backend system. When configuring the training via the web-interface, you can select your custom dataset by entering the name of the dataset.


### Cover and Phenology Training Dataset

In order to be able to use a custom cover and phenology image dataset, prepare your dataset by structuring it as described in [`plantcapnet_compute/datasets/dataset_structure.md`](plantcapnet_compute/datasets/dataset_structure.md), and then put it into the [`plantcapnet_compute/datasets/`](plantcapnet_compute/datasets/) subfolder. For more details, please check the respective README for the [PlantCAPNet_compute Backend](plantcapnet_compute/README.md).


### Training

Configure training using the web-frontend, and then run the provided command using the backend:

```bash
cd plantcapnet_compute/

python construct_ensemble.py <args>
```

The training can take some time, depending on the size of the datasets, number of classes (species), number of epochs trained and image resolutions used. For example, using an image dataset with 100.000 images with cover and phenology annotations with image resolutions of about 3200x1600 px and 40 epochs, the plant cover and phenology training alone can take 3-5 days on a NVIDIA A100 GPU. The duration of the pre-training is primarily dependent on the number of images and the number of classes. With about 75 classes and 600 training images per class, the entire pre-training process can take 2-4 days. The training durations scale approximately linearly, so this might help estimating the entire duration of a training process.

### Integration of New Model into the Web Interface

After a successful training, the script will copy the trained model(s) to [`plantcapnet_compute/output_models/`](plantcapnet_compute/output_models/) into a subfolder corresponding to your model run name. For inference usage of the model, copy  the resulting model (ideally with a descriptive name) into the `models/` folder. In addition to the model, copy the `class_names.txt` into the models folder under as `<model_name>_class-labels.txt` to make the web interface aware of the correct class names. E.g., if your model file is named `mymodel_60_species.pth`, then the class labels file should be named `mymodel_60_species_class-labels.txt`.

Furthermore, make a respective entry in the `config/models.yaml` file specifying some meta-information like default image resolution to be used during inference, prediction engine, preprocessing and model path to be able to use it with the interface. Images input to the model will be preprocessed with the provided preprocessing method (which is `torch` by default when using the backend), and will, by default, be resized to the provided input resolution. Note that for most configuration values there are already defaults set up, as described in the file itself. Hence, you primarily need to specify the model path, and check, if the default fit your needs and adjust otherwise.

An example entry can look like:

```yaml
cover-trained:
  InsectArmageddon:
    engine: Standardized
    model: models/ia_cover-trained.pth
    preprocessing: torch
    input_width: 2688
    input_height: 1536

zero-shot:
```

Use the **Standardized** engine (used by default), which covers most use cases. Further engines for specific needs can be implemented and set up as well.

In the case of zero-shot models, the entry should look the same, but should be specified in the "zero-shot" section.