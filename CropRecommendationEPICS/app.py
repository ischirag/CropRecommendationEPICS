
from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle
import requests

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)

def get_weather():
    api_key = "d850f7f52bf19300a9eb4b0aa6b80f0d"  # Free OpenWeatherMap API key
    city = "Bhopal"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity']
    }

@app.route('/')
def index():
    weather = get_weather()
    return render_template("index.html", weather=weather)

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    moisture = float(request.form.get('Moisture', 50))  # Default moisture 50%
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = mx.transform(single_pred)
    prediction = model.predict(scaled_features)

    crop_dict = {
        1: {"name": "wheat", "info": "Wheat is a grass widely cultivated for its seed. Requires moderate temperature and rainfall."},
        2: {"name": "soybean", "info": "Soybeans are legumes that need warm temperatures and good moisture levels."},
        3: {"name": "gram", "info": "Gram or chickpea is a drought-resistant crop that thrives in cool temperatures."},
        4: {"name": "paddy", "info": "Paddy (rice) requires high temperature, high humidity and rainfall of 100-200 cm."},
        5: {"name": "maize", "info": "Maize needs moderate temperature and grows well in well-drained soils."},
        6: {"name": "mustard", "info": "Mustard grows best in cool weather and well-drained soil."},
        7: {"name": "lentil", "info": "Lentils prefer cool temperatures and can grow in various soil types."},
        8: {"name": "urad", "info": "Urad dal requires warm temperature and moderate rainfall."},
        9: {"name": "groundnut", "info": "Groundnut needs warm climate and well-distributed rainfall."},
        10: {"name": "tomato", "info": "Tomatoes need warm weather and well-drained, fertile soil."}
    }

    if prediction[0] in crop_dict:
        crop_info = crop_dict[prediction[0]]
        result = f"{crop_info['name'].capitalize()} is the best crop to be cultivated right there"
        info = crop_info['info']
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        info = ""

    weather = get_weather()
    return render_template('index.html', result=result, crop_info=info, weather=weather)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
