# Meeting Request Classifier

Recommend using a virtual environment to run this code. To create a virtual environment, run the following command:

```bash
python3 -m venv venv
```

Then, activate the virtual environment:

```bash
source venv/bin/activate
```

Finally, to install the required packages, run:

```bash
pip install -r requirements.txt
```

Currently, the model is loaded via the `train.py` file, which you can run via `python train.py`. This will train the model and save the results to the `meeting_request_classifier.pkl` file (overwriting whatever is there).

## Running the Docker Container

This works in Docker; when the image is built, the model is trained and saved to the `meeting_request_classifier.pkl` file in the `docker` directory in the container. Running a container from this image with a volume mounted to the `docker` directory will allow you to access the results.

1. run `docker build -t ml-test .` (this creates the docker image, etc)
2. run `docker run -v <path-to-results>:/docker/results ml-test`
3. the results will be saved to the directory specified with `<path-to-results>`
