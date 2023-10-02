import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the data as a dictionary
kaggle_data = {
    "": [
        "GeoNuclearData",
        "GreaterManchesterCrime",
        "Pesticide",
        "StudentMathScore",
        "TheHistoryofBaseball",
        "USWildFires",
        "WhatCDHipHop",
        "WorldSoccerDataBase"
    ],
    "execution accuracy (%)": [100, 88.89, 85, 80, 90.91, 89.66, 95.65, 72.22],
    "valid score (%)": [68.75, 100, 68.00, 42.86, 51.28, 70.27,53.66, 72.22],
    # "incorrect sql2cypher (%)": [0, 11.11, 18.42, 20.00, 13.64, 3.70, 0, 14.29],
    "invalid sql2cypher (%)": [0, 0, 2, 0, 0, 0, 2.44, 0],
    "invalid parsed SQL (%)": [31.25, 0, 18, 46.43, 43.59, 21.62, 41.46, 0.0],
    
}

spider_data = {
    "": [
        "train",
        "development"
    ],
    "execution accuracy (%)": [82.83, 86.74],
    "valid score (%)": [51.23, 60.02],
    # "# incorrect sql2cypher": [897, 92]
    "invalid sql2cypher (%)": [6.79, 8.70],
    "invalid parsed SQL (%)": [38.16, 30.81],
}

for i, data in enumerate([kaggle_data, spider_data]):
    # Create a DataFrame from the data                               
    df = pd.DataFrame(data)
    if i==0:
        degree = 90
    else:
        degree=0

    # Set the figure size for the plots
    plt.figure(figsize=(15, 10))

    # Create subplots
    plt.subplot(2, 2, 1)
    sns.barplot(data=df, x="", y="execution accuracy (%)")
    plt.title("Valid Parsed SQL and Correctly SQL2Cypher Translation (%)")
    plt.xticks(rotation=degree)

    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x="", y="valid score (%)")
    plt.title("Valid Score (%)")
    plt.xticks(rotation=degree)

    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x="", y="invalid sql2cypher (%)")
    plt.title("Invalid SQL2Cypher Percentage")
    plt.xticks(rotation=degree)

    plt.subplot(2, 2, 4)
    sns.barplot(data=df, x="", y="invalid parsed SQL (%)")
    plt.title("Invalid Parsed Percentage")
    plt.xticks(rotation=degree)

    # Adjust layout
    plt.tight_layout()

    plt.savefig('{}.pdf'.format(i), format='pdf')

    # Show the plots
    plt.show()
