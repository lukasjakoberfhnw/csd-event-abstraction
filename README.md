# Installation

## Requirements

### 1. Clone the repository to your local machine.

### 2. Install Python 3.11 on the target machine

If Python 3.11 is already installed, you can skip this point. It may work with other Python 3 Versions, it has not been extensively tested.

### 3. Install all requirements

This can be done by creating a new Python environment for the project or directly by installing the packages globally on the operating system. We recommend to use a new Python environment using venv. Once activated, run the pip install command.

```
pip install -r requirements.txt
```

### 4. Request Access to Dataset
Request required datasets from the repository owner. Make sure the file structure of the data fits the one described below (Example: /data/tale-camerino/from_massimiliano/Log/...).

### 5. Preprocess Datasets using the prepared Script
Run the data preprocessing using the following script: _/src/preprocessing/tale_preprocessing.py_. This creates the _processed_ folder in the data directory.

```
python ./src/preprocessing/tale_preprocessing.py
```

### 6. Run Test Benchmark using a Decision Tree
To create the first benchmark, we have decided on a decision tree to serve as a lower boundary for the accuracy of the models. This guides us as a comparison with other algorithms and approaches. 

To run the decision tree benchmark, execute the following code from the console:

```
python ./src/decision_tree_prototype.py
```


# Code structure

The repository is split into the following directories to keep related files grouped together.

| **Folder** | **Description**                                                                                                   |
|----------|-----------------------------------------------------------------------------------------------------------|
| /.venv   | Used for virtual Python environment. Initialized with requirements.txt                                    |
| /data    | Holds the datasets in a defined structure to read them from code                                          |
| /output  | Folder used for file outputs like learned machine learning models or images                               |
| /src     | Contains the source code for the whole project                                                            |

# Data Sources

The datasets are not uploaded to github to reduce the size of the repository.

## Tale

We have two different Tale Datasets, one from the origial github repository, and one from Massimiliano directly. Both datasets are listed in the /data/tale-camerino folder.

### Tale Github

**Location**: /data/tale-camerino/from_github; **Files**: MRS.xes, running-mrs.xes

### Tale Massimiliano

**Location**: /data/tale-camerino/from_massimiliano

| **Type** | **Path**                                                                                                   |
|----------|-----------------------------------------------------------------------------------------------------------|
| DIR      | /Log                                                                                                      |
| DIR      | /Log/[ISO 8601: datetime string]/                                                                         |
| FILE     | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db3"                                   |
| DIR      | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db"                                    |
| DIR      | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db"/drone_1                            |
| DIR      | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db"/tractor_1                          |
| DIR      | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db"/tractor_2                          |
| DIR      | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db"/tractor_3                          |
| FILES    | /Log/[ISO 8601: datetime string]/[ISO 8601: datetime string] + "_0.db"/**.csv                             |


This dataset is much bigger than the one found on GitHub. It contains multiple runs with detailed log files in .csv files.

#### Run file structure

Each run has multiple files describing information:

- clock.csv
- closest_tractor.csv
- parameter_events.csv
- performance_metrics.csv
- rosout.csv
- tf.csv
- tractor_position.csv
- weed_position.csv

#### Robot files structure

Each robot has an individual subdirectory consisting of the following files:

cmd_vel.csv (all robots)
flight_data.csv (drone_1 only)
low_batter.csv (all robots)
macro.csv (all robots)
odom.csv (all robots)
range.csv (all robots)
tello_response.csv (drone_1 only)
weed_found.csv (drone_1 only)
blade_force.csv (tractors only)

The files could be explained more in detail with content and structure. However, we don't really know this yet.

File: Processing folder 2022-08-01_15_09_15.463175 (16/36) --> macro.csv is empty

#### 

## Other sources that might be interesting

- CASAS
- SCAND (https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH)

# Possible Approaches

Clustering:
- DBSCAN
- K-Means

Graph-based clustering:
- Temporal Event Graph Clustering
- Dynamic Bayesian Networks (DBN)
- Temporal Directed Acyclic Graphs (Temporal DAGs)
- Temporal Pattern Mining with Directed Graphs
- Hidden Markov Models (HMM) and Sequence Clustering
- Graph Neural Networks for Temporal Clustering
- Markov Chains, Bayesian Networks, Petri Nets
- Community Detection Algorithms (Louvain, Infomap)
- Spectral Clustering (Adjecency or Laplacian Matrix of the graph as input)

# Todo
Show the final classification in the raw-data for testing --> from which rows are the classes