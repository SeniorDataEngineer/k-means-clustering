# <a name="top-page"></a>KMeansClusteringPython
> Is an implementation of k-means clustering written from pseudo-code. The project includes a Jupyter
> Notebook illustrating usage of the package.

# Video demonstration
Download the file to view.
https://github.com/SeniorDataEngineer/k-means-clustering/blob/master/jer-kmeans-clustering.mp4

# Table of Contents
* [License](#license)
* [Team Members](#team-members)
* [Features](#features)
* [Project Structure](#structure)
* [Getting the project running](#run-project)
* [Standards](#standards)
* [Testing](#testing)

# <a name="license"></a>License
* None

# <a name="team-members"></a>Team Members
* "Mr. James Rose" <jameserose8@gmail.com>

# <a name="features"></a>Features
* K means clustering.
    * K means clustering for given k.
    * Calculating the silhouette coefficients.
    * Find best fitting k for the data set iteratively.
* Notebook.
    * Select k clusters and n vectors.
    * Manually cluster an generated data set.
    * Return silhouette scores for k selection.
    * Automatically find best fitting k.

# <a name="structure"></a>Project Structure
```
|_ .vscode
  |_ ... 
|_ Notebooks
  |_ k_means_clustering_demo.ipynb 
|_ src  
  |_ my_k_means
    |_ __init__.py 
    |_ k_means_clustering.py
  |_setup.py
|_ .gitignore
|_ README.md
|_ requirements.txt
```

# <a name="run-project"></a>Getting the project running
This guide assume that you are using Visual Studio Code and Git.

1. [Download](https://www.anaconda.com/products/individual) & install Anaconda distribution of Python 3.8.5.
1. Create a virtual environment called **venv-k-means-clustering** using Anaconda Navigator at _C:\Users\username\anaconda3\envs_ .
1. Clone project to _C:\Users\username\source\repos_ on local machine.
1. Install package into **venv-k-means-clustering**.
1. Open project workspace in VSC.
1. Open terminal and `pip install -r requirements.txt`.
1. Open the notebook and run cells.

# <a name="standards"></a>Standards
All code has been checked with `pycodestyle` utility to match  
**pep8** guidance. 
## Usage of `pyscodestyle`
Navigate to directory where the file you want to test is located  
using `cd` in terminal. Run `pycodestyle filename.py` and wait  
for output in terminal. If no output standard is met.

# <a name="testing"></a>Testing
## Unit testing
The unit tests for the package are stored in the python modules
as doctests. When running a module standalone the doctests will
be called for that module.  
To run tests; navigate to *src/my_k_means/* and run command  
`python -m filename.py`.

-------
[Return to top](#top-page)
