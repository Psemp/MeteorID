import mlflow
import os
import pickle
import json

from matplotlib import pyplot as plt


def train_and_log_keras(
        model, model_summary, history, mtype_list: list, metrics: dict,
        model_name: str = "cnn_meteorite_clf", dataset_version: str = "pre_release"
        ):
    """
    Trains and predict the model based on X_train/test and y_train/test + classifier (keras DNN)
    Logs the model name (default is dnn_classifier) using mlflow as well as the params and
    the metrics (accuracy, f1, recall) using mlflow. Handles description creation with json template

    Args :
    - model : keras model, fitted
    - model_name : default = cnn_meteorite_clf, the name of the model as will appear on mlflow's logs
    - dataset_version : default = pre_release, the version of the dataset
    - feature list : the list of features used by the model

    Returns :
    - metrics : dictionnary of the evaluated metrics
    - model : the classifier fitted on X_train/y_train and params
    """

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    description = describe_run(
        template_path="../templates/description_mlflow.json",
        model_name=model_name,
        data_version=dataset_version,
        )

    with mlflow.start_run(run_name=model_name, description=description):

        # Metrics :
        accuracy = metrics["accuracy"]
        f1 = metrics["f1"]
        recall = metrics["recall"]

        metrics = {"accuracy": accuracy, "f1": f1, "recall": recall}

        # MLFlow log :
        mlflow.log_metrics(metrics)

        with open("model_summary.txt", "w") as f:
            f.write(model_summary)

        serialized_list = pickle.dumps(mtype_list)
        with open("mtype_list.pkl", "wb") as f:
            f.write(serialized_list)

        mlflow.log_artifact("mtype_list.pkl")
        mlflow.log_artifact("model_summary.txt")

        epoch_fig = plot_epochs(history=history)

        mlflow.log_figure(epoch_fig, "epoch_eval_fig.png")

        model_type = type(model)

        if model_type.__module__.startswith("keras"):
            artifact_path = "keras-model"
            mlflow.tensorflow.log_model(model=model, artifact_path=artifact_path, registered_model_name=model_name)

        else:
            pass

        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/{artifact_path}"

        mlflow.register_model(model_uri=model_uri, name=model_name)

        return metrics, model


def describe_run(
        template_path: str, model_name: str, data_version: str,
        ) -> str:
    """
    Function :
    Formats the template passed as argument and returns a description to be added to the mlrun

    Args :
    - template_path : path to the json template
    - model_name : the name of the model
    - data_version : the version of the dataset used
    - imb_learn_method : the method used to solve class imbalance, if any
    - column_drop_na_threshold : the threshold used to delete variables if na ratio > thresh

    Returns :
    - description : the description ready to be passed to mlflow to describe the run, as str
    """

    with open(template_path, "r") as template:
        template = json.load(template)

    description = template["description"].format(
        model_name=model_name,
        data_version=data_version,
    )

    return description


def plot_epochs(history):
    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        figsize=(8, 4),
        dpi=150
        )

    plt.ioff()

    training_recall = history.history["recall"]
    validation_recall = history.history["val_recall"]
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    ax1.plot(training_recall, label="training recall")
    ax1.plot(validation_recall, label="validation recall")

    ax2.plot(training_loss, label="training loss")
    ax2.plot(validation_loss, label="validation loss")

    ###
    # Titles/Lables
    ax1.legend()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Recall")
    ax2.legend()
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    fig.suptitle("Evolution of Training and Validation accuracy per epoch")
    #
    ###

    return plt.gcf()
