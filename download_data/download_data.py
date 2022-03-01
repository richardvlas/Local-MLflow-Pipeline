#!/usr/bin/env python
import os
import argparse
import logging
import pathlib
import requests
import tempfile
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    # Derive the base name of the file from the URL
    basename = pathlib.Path(args.file_url).name.split("?")[0]

    # Download file, streaming so we can download files larger than
    # the available memory. We use a named temporary directory that gets
    # destroyed at the end of the context, so we don't leave anything
    # behind and the file gets removed even in case of errors
    logger.info(f"Downloading {args.file_url} ...")
    with tempfile.TemporaryDirectory() as temp_dir:

        logger.info("Creating run")
        # Launch a run. The experiment is inferred from the 
        # MLFLOW_EXPERIMENT_NAME environment
        with mlflow.start_run() as run:

            path = os.path.join(temp_dir, args.artifact_name)
            
            # Set name of the run as a tag
            mlflow.set_tag("mlflow.runName", "download_data")

            # Download the file streaming and write to open temp file
            with open(path, mode='wb+') as fp:
                with requests.get(args.file_url, stream=True) as r:
                    for chunk in r.iter_content(chunk_size=8192):
                        fp.write(chunk)

                # Make sure the file has been written to disk before uploading
                # to mlflow as file artifact
                fp.flush()

            logger.info("Logging artifact")
            mlflow.log_artifact(path)

            # Set artifact tags
            logger.info("Logging tags")
            tags = {
                "artifactName": args.artifact_name,
                "artifactType": args.artifact_type,
                "artifactDescription": args.artifact_description,
                "artifactOriginalUrl": args.file_url
            }
            mlflow.set_tags(tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to MLflow", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
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
