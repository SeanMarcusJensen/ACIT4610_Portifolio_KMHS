# ACIT4610 Evolutionary Intelligence : Problem 4

---

## How to run

All the agents are pre-run / trained. You can find all their weights in the folder `static/weights`.
The parameters of the models are found though the script `utils/parameter_tuner.py`. If you want to find the best parameters again then you're free to run this script by `python -m utils.parameter_tuner`.

The notebook `main.ipynb` contains all the code for running the agent live, visualizing the metrics, and even training the agents again; The training process is placed at the bottom of the notebook to allow the reader to quickly read and see the agents in action.
This is also because some of the agents take a really long time to train, and to not force the reader to invest much time or money in GPUS.

**NOTE**: To run the agents and the notebooks yourself, some libraries are required:

You can see them all in the `requirements.txt` file, and install them by `pip install -r requirements.txt`. [Please be inside a python environment]

- Numpy
- Pandas
- Tensorflow[and-cuda]
- Matplotlib
- Optuna
- Gymnasium
- Pillow
