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


# Different Preprocessing Approaches

We have developed different preprocessing algorithms to find the method that leads to the best performance.

## Preprocessing 1: Forward fill X, Y, Z

This approach was used to test the hypothesis if we could simply join the .csv files and forward fill the missing values using the ffill function provided by the pandas library. This method first combines all csv files into one massive table and fills missing X, Y, Z values based on the last valid value. One crucial problem with this approach is the missing variability within the positional arguments. When the position is always just filled, it does not incorporate the variability and does not correctly merge. 

## Preprocessing 2: Forward fill activities
Similar to the previous method, this function propagates the activities instead of the positional variables. This ensures the full data scale using all positional rows. However, the inconsistencies within the dataset caused inappropriate value propagation. While this method gets better results, deeper analysis of the results shows wrongful alignments between events and positional values. 

## Preprocessing 3: Lifecycle transformation

This method relies on a similar approach to an already existing code base provided by our supervisor Massimiliano Sampaolo. It processes the macro.csv file using all events and fills them into the odom.csv file using timestamps as orientation. Events always are identified with a START and STOP lifecycle flag. This algorithm fills the selected event from the START timestamp until the time indicated within the STOP row. Highlighting the inconsistencies in the data, some runs were not processable using this method because of the misalignment of START and STOP values within the macro.csv file. 

## Preprocessing 3.5: Adjusted Lifecycle transformation

This preprocessing method integrates feature engineering creating the following variables: dx (difference in x in relation to the previous row from the same robot), dy (difference in y in relation to the previous row from the same robot), dz (difference in z in relation to the previous row from the same robot), has_payload (a transformation scheme that checks for certain distinctions such as “weed”, “/tractor”, or “name” within the payload string, and creates dummies for the categorical columns (robot and has_payload) using the get_dummies function from the pandas library. 

## Preprocessing 4: Manually engineered features

Opposed to the previous methods, this algorithm was created directly from the manual data analysis and merges the different .csv files as a first step. Afterwards, it forward fills the positional values and calculates the various difference variables: dx (difference in x in relation to the previous row from the same robot), dy (difference in y in relation to the previous row from the same robot), dz (difference in z in relation to the previous row from the same robot). Next, it propagates specific events that are supposed to last longer than a split second: EXPLORE, TAKEOFF, LAND, MOVE, and CUT_GRASS. This propagation is done with the help of the lifecycle column within the macro.csv. It transforms the payload from a continuous string into a category by applying a function checking for a specific substring such as “weed”, “/tractor”, or “name”. Finally, the algorithm fills the not known events with IDLE and adds a column minutes_since_start for indicating the current time in the simulation using minutes. 

# Results

## Baseline with Decision Tree Classifier

| Preprocessing Approach                     | Precision | Recall | F1-Score | Support  |
|--------------------------------------------|-----------|--------|----------|----------|
| 1: Forward fill with X, Y, Z values       | 0.15      | 0.11   | 0.11     | 650,714  |
| 2: Forward fill with activities           | 0.43      | 0.39   | 0.37     | 506,470  |
| 3: Lifecycle transformation               | 0.27      | 0.35   | 0.27     | 650,641  |
| 3.5: Adjusted lifecycle transformation    | 0.69      | 0.65   | 0.65     | 650,641  |
| 4: Manually engineered features           | 0.72      | 0.64   | 0.66     | 650,714  |

## Random Forest Classifier

| Preprocessing Approach                     | Precision | Recall | F1-Score | Support  |
|--------------------------------------------|-----------|--------|----------|----------|
| 1: Forward fill with X, Y, Z values       | 0.10      | 0.10   | 0.10     | 650,714  |
| 2: Forward fill with activities           | 0.41      | 0.41   | 0.39     | 506,470  |
| 3: Lifecycle transformation               | 0.28      | 0.37   | 0.31     | 650,641  |
| 3.5: Adjusted lifecycle transformation    | 0.76      | 0.66   | 0.67     | 650,641  |
| 4: Manually engineered features           | 0.76      | 0.64   | 0.66     | 650,714  |

## AdaBoost Classifier

| Preprocessing Approach                     | Precision | Recall | F1-Score | Support  |
|--------------------------------------------|-----------|--------|----------|----------|
| 1: Forward fill with X, Y, Z values       | 0.10      | 0.10   | 0.10     | 650,714  |
| 2: Forward fill with activities           | 0.21      | 0.27   | 0.23     | 506,470  |
| 3: Lifecycle transformation               | 0.11      | 0.12   | 0.12     | 650,641  |
| 3.5: Adjusted lifecycle transformation    | 0.30      | 0.21   | 0.24     | 650,641  |
| 4: Manually engineered features           | 0.28      | 0.11   | 0.12     | 650,714  |

## Multi-Layer Perceptron

| Preprocessing Approach                     | Precision | Recall | F1-Score | Support  |
|--------------------------------------------|-----------|--------|----------|----------|
| 1: Forward fill with X, Y, Z values       | 0.13      | 0.11   | 0.11     | 650,714  |
| 2: Forward fill with activities           | 0.31      | 0.38   | 0.33     | 506,470  |
| 3: Lifecycle transformation               | 0.19      | 0.26   | 0.20     | 650,641  |
| 3.5: Adjusted lifecycle transformation    | 0.21      | 0.33   | 0.22     | 650,641  |
| 4: Manually engineered features           | 0.57      | 0.45   | 0.46     | 650,714  |

## Recurrent Neural Network

| Preprocessing Approach                     | Precision | Recall | F1-Score | Support  |
|--------------------------------------------|-----------|--------|----------|----------|
| 4: Manually engineered features           | 0.87      | 0.78   | 0.77     | 93050  |

# Final remarks

This project was done in the scope of the Complex System Design module at the University of Camerino. For further questions towards the code or the data, please do not hesitate to reach out to me.