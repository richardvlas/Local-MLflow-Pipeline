#!/usr/bin/env python
import os
import argparse
import logging
import seaborn as sns
import pandas as pd
import tempfile
import mlflow
from  mlflow.tracking import MlflowClient
from sklearn.manifold import TSNE


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

client = MlflowClient()

def go(args):

    with mlflow.start_run() as run:
        # Search all runs under experiment id and order them by
        # descending value of the start_time       
        exp_id = run.info.experiment_id
        filter_string = f"tags.artifactName='{args.input_artifact}'"
        order_query_string = "attribute.start_time DESC"
        runs = client.search_runs(exp_id, 
                                  filter_string=filter_string,
                                  order_by=[order_query_string])
        run_of_input_artifact = runs[0]

        artifact_path = os.path.join(run_of_input_artifact.info.artifact_uri,
                                     args.input_artifact)
        iris = pd.read_csv(
            artifact_path,
            skiprows=1,
            names=("sepal_length", "sepal_width", "petal_length", "petal_width", "target")
        )

        target_names = "setosa,versicolor,virginica".split(",")
        iris["target"] = [target_names[k] for k in iris["target"]]

        logger.info("Performing t-SNE")
        tsne = TSNE(n_components=2, init="pca", random_state=0)
        transf = tsne.fit_transform(iris.iloc[:, :4])

        iris["tsne_1"] = transf[:, 0]
        iris["tsne_2"] = transf[:, 1]

        g = sns.displot(iris, x="tsne_1", y="tsne_2", hue="target", kind="kde")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "t-SME.png")
            g.savefig(image_path)

            logger.info("Uploading image to MLflow")
            mlflow.log_artifact(image_path)

            logger.info("Creating cleaned data artifact")
            data_path = os.path.join(temp_dir, "clean_data.csv")
            iris.to_csv(data_path)

            logger.info("Logging artifact")
            mlflow.log_artifact(data_path)

            # Set artifact tags
            logger.info("Logging tags")
            tags = {
                "inputArtifact": args.input_artifact,
                "artifactName": args.artifact_name,
                "artifactType": args.artifact_type,
                "artifactDescription": args.artifact_description,
            }
            mlflow.set_tags(tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process an input artifact/file and upload it as an artifact to mlflow",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
