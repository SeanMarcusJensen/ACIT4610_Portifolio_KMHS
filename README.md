# ACIT4610_Portifolio_KMHS

## How to setup & run projects?

Each problem folder has one `environment.yml` file. This file contains everything you need to run the code / problem.
The `environment.yml` file is used by `Miniconda` / `conda` to create the environment with all packages and even the environment name we use to separate the individual problems. This is used to choose both correct python version and packages, such that we minimize hassle under development. Thus, installing `Miniconda` / `conda` is a **PREREQUISITE** to running and using the codebase.

> Installation docs if not already installed*: [Miniconda - Installation Guide](https://docs.anaconda.com/miniconda/).

### How to manage environments?

>Conda has an amazing guide on how to manage environments. *[Conda Documentation - Managing Environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)*

That said, I'll outline the most important and frequent usages below!

### How to setup the environment?

So, you're going to start coding or just want to explore the code yourself. The first thing you'll need is to be able to run the code, but how do you do that?

Here is a step by step guide on how to setup the environment:

**1. Open terminal.**

Make sure to be inside the terminal on your computer.
If you do not know what a terminal is, or do not know how to do so in the editor og on your computer, please go to google and search for it. e.g [How to open terminal](https://www.google.com/search?q=How+to+open+terminal).

---

**2. Be inside the correct folder.**

You need to be inside the correct folder path. You can check where you are in terminal by running the command: `pwd`.
Make sure the output is `{YOUR_PATH_TO_PROJECT}/ACIT4610_Portifolio_KMHS/{PROBLEM_TO_RUN}`

If you are only inside the `{YOUR_PATH_TO_PROJECT}/ACIT4610_Portifolio_KMHS` then please proceed to run `cd {PROBLEM_TO_RUN}`, e.g. `cd problem_1`.

---

**3. Create the environment on your computer**

Direct docs [Creating an environment from an environment.yml file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

In the terminal please run the command:

```sh
conda env create -f environment.yml
```

This will create the environment on your computer with the name we have choosen for the current problem. The reason to have the same name for the problem for everyone is because of `Merge Conflicts` in `GitHub` because of `Jupyter Nootebook Metadata`. And to maintain full Transparancy and fluent team work.

Make sure the correct conda environment is active

---

**4. Your environment should now be created.**

The `conda` environment should now be created. You are now able to use the environment in the `python` & `Jupyter Notebooks`.

> Each problem will now outline how to run that problem. Please proceed to the respected README.md file.

---

### How to install new packages to the environment?

So, you have a package you need to install into the environment. You can just use `pip install` like you'd normally do. **But make sure to update the `environment.yml` file.**

**NB - But make sure you are in the `{YOUR_PATH_TO_PROJECT}/ACIT4610_Portifolio_KMHS/{PROBLEM_TO_RUN}` path.**

Use this command:

```sh
conda env update --file environment.yml --prune
```

---

## Standards to follow

### Python

For the code, we'll use the PEP 8 standard [https://peps.python.org/pep-0008/], one exclusion is that we're using tabs instead of spaces.

We should also use python version `3.11` - because not all libraries support versions above `3.11`. Such as `Tensorflow`.

#### Documentation

Every python class should have a short description stating the intention of the class and use-case. If the docstring is too large, it is probably an indicator to that the class is too big and should be split into multiple classes.

[**READ MORE ABOUT DOCS BY CLICKING HERE**](https://developer.lsst.io/v/DM-5063/docs/py_docs.html)

```python
class TestClass:
    """ This is a short description of the TestClass.
        Is has no other use-case than being an example.
    """
    def __init__(self) -> None:
        pass
```

### Git Strategy

We follow (Gitflow)[https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow] for structuring branches and such.

Whenever solving an **Issue**, create a new branch out from **main**, call it **feature/{issue_id}-{some-good-title}**, where the *{issue_id}* is the **#** in the Issue title, and the title is something meaningful.
Once the **Issue** is done, and the feature branch is ready for merge, create a **Pull Request** into **main**, and make sure someone approves the changes made.

### Folder Structure

Each folder at the `root` of the project should represent `ONE` problem from the exam paper. Meanwhile, each problem-folder **MUST** cointain atleast:
- **README.MD** file with instructions on how to run the problem code and an explaination of the approach used: such as preprocessing, etc. (*You can see the requirements for this file in the exam paper*).
- **environment.yml** file is the environment to install to be able to use the code.
- **data**-folder: This is where we will store all our data files used in the project, if any! Such as CSV & JSON files.
    - could be useful to separate out **raw data** and **pre-processed data** into different folders for better structure.
- **main.ipynb or main.py**-file: This will be the entry point for the solution to the problem at hand.
- we should also keep the folders organized. Therefore, if we have any *utility code* or whatnot, we should place it in a **utils** folder. Try to keep the code as separate as possible, making the **main file** as concise as possible while maintaining readability etc.
- **test**-folder is where we store all test files.
