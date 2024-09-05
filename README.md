# ACIT4610_Portifolio_KMHS

## Standards to follow

### Python

For the code, we'll use the PEP 8 standard [https://peps.python.org/pep-0008/], one exclusion is that we're using tabs instead of spaces.

### Git Strategy

We follow (Gitflow)[https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow] for structuring branches and such.

Whenever solving an **Issue**, create a new branch out from **main**, call it **feature/{issue_id}-{some-good-title}**, where the *{issue_id}* is the **#** in the Issue title, and the title is something meaningful.
Once the **Issue** is done, and the feature branch is ready for merge, create a **Pull Request** into **main**, and make sure someone approves the changes made.

### Package Management

All packages used for the specific problem **MUST** be printed into a *requirements.txt* file. This can be done by using the `pip freeze > requirements.txt` command!
**NB - MAKE SURE YOU HAVE ACTIVATED YOUR PYTHON ENVIRONMENT FIRST!** - *see below*

### Environments

We must always use a local environment when working on the codebase.
Each problem-folder contains a **requirements.txt** file which holds all of the required libraries used for that specific problem.
To install the required packages: `pip install -r requirements.txt`

### Folder Structure

Each folder at the `root` of the project should represent `ONE` problem from the exam paper. Meanwhile, each problem-folder **MUST* cointain atleast:
- **README.MD** file with instructions on how to run the problem code and an explaination of the approach used: such as preprocessing, etc. (*You can see the requirements for this file in the exam paper*).
- **data**-folder: This is where we will store all our data files used in the project, if any! Such as CSV & JSON files.
    - could be useful to separate out **raw data** and **pre-processed data** into different folders for better structure.
- **main.ipynb or main.py**-file: This will be the entry point for the solution to the problem at hand.
- we should also keep the folders organized. Therefore, if we have any *utility code* or whatnot, we should place it in a **utils** folder. Try to keep the code as separate as possible, making the **main file** as concise as possible while maintaining readability etc.

