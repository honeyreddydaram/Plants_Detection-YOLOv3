from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Define a flask app
app = Flask(__name__)

app.static_folder = 'static'
MODEL_PATH ='engine/best.pt'

model = YOLO(MODEL_PATH)

description={'Oleander':"""
Plant: oleander -Nerium(గన్నేరు)
Family:dogbane family (Apocynaceae)
Origin:Northwest Africa and Iberian
Uses:heart conditions, asthma, epilepsy, cancer, painful menstrual periods, leprosy, malaria, ringworm, indigestion, and venereal disease.
How to Use:The oleander plant contains a number of related cardiac glycosides similar in activity to digitalis.
             

""",'Azadiractha Indica':"""
Plant: Neem-Azadirachta indica (వేప)
Family:Meliaceae
Origin: Indian subcontinent and to parts of southeast Asia 
Uses: Neem preparations are reportedly efficacious against a variety of skin diseases, septic sores, and infected burns.
How to use: You can clean some fresh leaves with water. Grind the leaves and add water to it to make neem juice. You may consume raw neem juice to partake in its health effects.

""",'Ficus Religiosa-Raavi-':"""
Plant:Ficus religiosa - sacred fig(రావి చెట్టు)
Family:Moraceae
Origin:Indian subcontinent
Uses:asthma, diabetes, diarrhea, epilepsy, gastric problems, inflammatory disorders, infectious.
How to use: (1)Take 5-10ml Peepal leaves juice or as directed by the physician.
Mix with mild hot water have it before going to bed.
To get rid of the constipation.
(2)Take 2-4gm of Peepal bark powder.
 Boil it 1 cup of water 10 minutes or water remain ¼ in quantity.
 Stain it and have it 15ml -20ml twice a day or as directed by the physician.
To get rid of the symptoms of diarrhea.

""",'Calotropis':"""
Plant: Calotropis- milkweed(జిల్లేడు)
Family:Apocynaceae family
Origin:southern Asia and North Africa
Uses:digestive disorders, toothache, cramps, joint pain.The milky juice of Calotropis procera was used against arthritis, cancer, and as an antidote for snake bite.
How to use:A potent herb that can be made into powder and oil, Calotropis gigantea can cure many health disorders if applied topically or inhaled. However the doctor's suggestion is important.
"""}



def model_predict(file, model):
    results = model.predict(file,stream=True)
    names = model.names
    for r in results:
        for c in r.boxes.cls:
            label=names[int(c)]
    return label


@app.route('/', methods=['GET'])
def index():
    return render_template('U.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
             basepath, 'static/upload', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
    return render_template('U.html',prediction_text="Type is {}".format(preds),details=description[preds],filename=f.filename)


if __name__ == '__main__':
    app.run()