from flask import Flask,request,jsonify
import numpy as np
import pickle
model = pickle.load(open('model1.pkl','rb'))
appy = Flask(__name__)
@appy.route('/')
def index():
    return "Hello world"
@appy.route('/predict',methods=['POST','GET'])
def predict():
    cgpa = request.args.get('cgpa')
    iq = request.args.get('iq')
    profile_score = request.args.get('profile_score')
    input_query = np.array([[cgpa,iq,profile_score]])
    result = model.predict(input_query)[0]
    print(input_query)
    return jsonify({'placement':str(result)})
if __name__ == '__main__':
    appy.run(host=('0.0.0.0'), port=(80), debug=True, use_reloader = False)