#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
with open('model_pkl_gradientBoosting', 'rb') as files:
	model = pickle.load(files)
#model = pickle.load(open('model_pkl_gradientBoosting', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_features[0][4] = final_features[0][4]/10
    final_features[0][5] = final_features[0][5]/10
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    return render_template('index.html', prediction_text='Estimated Effective Young Modulus is :{}'.format(output*10))

if __name__ == "__main__":
    app.run(debug=True)