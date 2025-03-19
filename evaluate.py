from math import sqrt

import pandas as pd
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             precision_recall_fscore_support)


# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


if __name__ == "__main__":

    val_df = pd.read_csv("data/validation.csv")

    pred_df = pd.read_csv("validation_prediction.csv")

    df = pd.merge(
        val_df,
        pred_df,
        how="left",
        left_on=["ReviewerID", "ProductID"],
        right_on=["ReviewerID", "ProductID"],
    )
    print(df.shape)
    df.fillna(0, inplace=True)

    print("VALIDATION RMSE: ", rmse(df["Star_y"].values, df["Star_x"].values))
