import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pm4py
import re

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
output_path = os.path.join(os.path.dirname(__file__), '..', 'output')
tale_XES_file_location = os.path.join(data_path, 'tale-camerino', 'MRS.xes')

def main():
    print("Visualization")
    # create a dataframe with three features of type (int, float, string)
    test_dataframe = pd.DataFrame({'int_feature': [1, 2, 3, 4, 5],
                                   'float_feature': [1.1, 2.2, 3.3, 4.4, 5.5],
                                   'string_feature': ['a', 'b', 'c', 'd', 'e']})
    
    # get actual data from the xes file
    log = pm4py.read_xes(tale_XES_file_location)
    log_df = pm4py.convert_to_dataframe(log)

    # drop column with many unique values where column type is object 
    categorical_values = log_df.select_dtypes(include=[object]).columns
    # drop columns with many unique values
    n_unique_values = log_df[categorical_values].nunique()
    columns_to_drop = n_unique_values[n_unique_values > 50].index
    log_df = log_df.drop(columns=columns_to_drop)

    visualize(log_df)

def visualize(features: pd.DataFrame, show_numerical_by_category=False):
    print("Visualize")
    pattern = re.compile(r'^datetime.*')

    # get the datatype of the features into a list
    feature_types = features.dtypes
    print(feature_types)

    categorical_features = []
    numerical_features = []
    time_features = []

    # iterate over the features and add them to the respective list
    for feature in features.columns:
        # if datetime is a feature, convert it to a string
        print(feature_types[feature])
        if feature_types[feature] == 'datetime64[ns, UTC]':
            print("Converting datetime to string")

            features[feature] = features[feature].astype(str)
            time_features.append(feature)
            continue
        if feature_types[feature] == 'object':
            categorical_features.append(feature)
        else:
            numerical_features.append(feature)

    print(categorical_features)
    print(numerical_features)
    print("Features correctly separated")

    for feature in numerical_features:
        plt.hist(features[feature])
        plt.title(feature)
        plt.show()

    for feature in categorical_features:
        features[feature].value_counts().plot(kind='bar')
        plt.title(feature)
        plt.show()

    # use the category to further describe the numerical features
    if show_numerical_by_category:
        for feature in numerical_features:
            for category in categorical_features:
                # create plot for each category
                features.boxplot(column=feature, by=category)
                plt.show()

    # create a scatter plot for each numerical feature colorized by the categorical feature
    for feature in numerical_features:
        for category in categorical_features:
            features.plot.scatter(x=feature, y=category)
            plt.show()

    # Create scatter plots --> many many
    for num_feature in numerical_features:
        for cat_feature_x in categorical_features:
            if not cat_feature_x == "concept:name":
                continue
            for cat_feature_color in categorical_features:
                if cat_feature_color == cat_feature_x:
                    continue
                plt.figure(figsize=(8, 6))
                
                # Scatter plot
                scatter = plt.scatter(
                    x=features[cat_feature_x],
                    y=features[num_feature],
                    c=pd.factorize(features[cat_feature_color])[0],  # Color encoded as integers
                    cmap='viridis',  # Color map for better visualization
                    s=50,  # Marker size
                    alpha=0.7  # Transparency
                )
                
                # Add labels and title
                plt.colorbar(scatter, label=cat_feature_color)  # Legend for color
                plt.title(f"Scatter Plot: {num_feature} by {cat_feature_x} (colored by {cat_feature_color})")
                plt.xlabel(cat_feature_x)
                plt.ylabel(num_feature)
                plt.grid(True, linestyle='--', alpha=0.7)

                # Show the plot
                plt.show()


if __name__ == '__main__':
    main()