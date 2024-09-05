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

