# ACIT4610 Evolutionary Intelligence : Problem 4

---

## How to run

All the agents are pre-run / trained. You can find all their weights in the folder `static/weights`. However, If you want to run the training loop yourself, proceed to run the `train.py` file at the root of `problem_4`. Make sure to have the Python environment with the required packages installed activated in the terminal before running. This will train all the agents and save a copy of the metrics, weights and a movie-clip of 2 episodes in the static folder. These files are then presented in the `main.ipynb` file.

If you do not want to run the models yourself, especially the deep-q-network ( which takes a lot of time ), proceed to open the `main.ipynb` file. The rest of the documentation should be described in markdown there.

**NOTE: Computer Specs:**
The deep-q-model are run on a NVIDIA 4080 with over 10_000 cuda cores.

CPU: AMD Ryzen 9 3XD - Mhz....

These specs is important in the analysis of the elapsed time for each episode.

