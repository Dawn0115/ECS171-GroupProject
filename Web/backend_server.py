from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

df_train = pd.read_csv("train_cleaned_sex_binary.csv")
freq_shell = df_train["Shell weight"].value_counts(normalize=True).to_dict()
freq_whole = df_train["Whole weight"].value_counts(normalize=True).to_dict()

def manual_openfe_features(df_raw):
    df = df_raw.copy()

    freq_feature_shell = df["Shell weight"].map(freq_shell).fillna(0)
    freq_feature_whole = df["Whole weight"].map(freq_whole).fillna(0)

    df_manual = pd.DataFrame({
        "f01_Length_div_ShellWeight": df["Length"] / df["Shell weight"],
        "f02_Whole1_div_ShellWeight": df["Whole weight.1"] / df["Shell weight"],
        "f03_Diameter_div_ShellWeight": df["Diameter"] / df["Shell weight"],
        "f05_Length_minus_Shell": df["Length"] - df["Shell weight"],
        "f07_freq_ShellWeight": freq_feature_shell,
        "f08_Max_Whole2_Shell": df[["Whole weight.2", "Shell weight"]].max(axis=1),
        "f09_log_Whole_weight": np.log(df["Whole weight"]),
        "f10_freq_WholeWeight": freq_feature_whole,
        "f11_Shell_plus_Height": df["Shell weight"] + df["Height"],
    })

    return df_manual

model = joblib.load("xgb_best_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw = pd.DataFrame([data])

    manual = manual_openfe_features(raw)
    X = pd.concat([raw.reset_index(drop=True), manual], axis=1)
    X = X.drop(columns=["Length", "Whole weight"], errors='ignore')

    prediction = model.predict(X)[0]
    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
