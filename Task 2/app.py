import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    hours=float(request.form.get('hours'))
    score=float(request.form.get('goal'))
    
    prediction = model.predict(np.array(hours).reshape(-1,1))
    output = round(prediction[0], 2)
    text = 'Predicted Student score is {} for a study of {} hours daily'.format(output,hours)

    compare = score-output
    message=""
    if compare>0:
        message="You need to work harder to reach your goal... All the Best"
    else:
        message="You will reach your goal with this practice... Well Done"
    
    

    return render_template('index.html', prediction_text=text,msg=message)


if __name__ == '__main__':
    app.run(debug=True)
