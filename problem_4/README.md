# ACIT4610 Evolutionary Intelligence : Problem 4

---

## How to Run

The project is in the `main.ipynb` file. This is a Jupyter Notebook which holds markdown text explaining the environment and python code for showcasing the project with graphs, tables etc.
All agents can be ran inside the `main.ipynb` file and at the bottom of the file is the code for training the agents. Do note that all agents are pre-trained with weights stored in `static/weights`. This is also where the `csv` files for hyperparameter tuning is stored.

The reason that all agents are pre-trained is because the Deep Q-Network Agent takes a very long time to train, even on a very good GPU: the computer specs on which the models are trained is in the top of the `main.ipynb` file. This matters because we do compare time per episode for each agent, and every agent except Deep Q runs on the CPU. Thus, time differences may vary, and especially if the reader do train the agents them self and then do the analysis again.

As mentioned, we do hyperparameter tuning, this script is `find_hyperparameters.py` and saves the output in `static/weights`. If you do want to run a tune yourself, execute `python tune_hyperparameters.py` -> **NOTE**: This will `pip install optuna` if you do not already have it installed.

**NOTE**: To run the agents and the notebooks yourself, some libraries are required:

You can see them all in the `requirements.txt` file, and install them by `pip install -r requirements.txt`. [Please be inside a python environment]

- Numpy
- Pandas
- Matplotlib
- Gymnasium[toy-text]
- Tensorflow

For training on the gpu:

- Tensorflow[and-cuda]

For parametertuning:

- Optuna [only downloaded when running hyperparam file.]
