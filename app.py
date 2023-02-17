import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = scaler.fit_transform([input_features])
    
    features_name = [ "age", "trestbps","chol","thalach", "oldpeak", "sex_0",
                       "sex_1", "cp_0", "cp_1", "cp_2", "cp_3","fbs_0","fbs_1",
                        "restecg_0","restecg_1","restecg_2","exang_0","exang_1",
                        "slope_0","slope_1","slope_2","ca_0","ca_1","ca_2","ca_3",
                     	"ca_4","thal_0","thal_1","thal_2","thal_3"]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "heart disease."
    else:
        res_val = "no heart disease."
        

    return render_template('index.html', prediction_text='Patient has {} And the accuracy of the result is {:.2f}%'.format(res_val, output.item() * 100))

if __name__ == "__main__":
    app.run(debug=True)