from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('fruit_sorting_model.h5')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    file_path = ''
    if request.method == 'POST':
        f = request.files['image']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        result = model.predict(img_array)
        prediction = 'Rotten' if result[0][0] > 0.5 else 'Fresh'

    return render_template('index.html', prediction=prediction, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
