from flask import send_file
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re
from HTTR import httr_prediction
from PdfToImage import Pdf2Image
from PilToCv import Pil2cv_converter
from ComputerVision import computer_vision_soft_version ,computer_vision_scanned_version
from Asag import pridector
from flask import send_from_directory
import pandas as pd

# Create flask app
flask_app = Flask(__name__)

Httr_model = pickle.load(open("model.pkl", "rb"))
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
Asag_model = SentenceTransformer("Zingy_modeling", device='cpu')



@flask_app.route("/")
def Home():
    return render_template("index.html")





@flask_app.route("/predict", methods = ["POST"])
def predict():
    
    torch.cuda.empty_cache()

    #dimensions
    imgs_dim = Pdf2Image(inputtype=0, path="Form_for_dimensions.pdf")     
    imgs_dim=[Pil2cv_converter(img_dim) for img_dim in imgs_dim]
    answerss, countorss, heightss, widthss = computer_vision_soft_version(imgs_dim[0]) 


    # Access text from the textarea
    model_answer_text = request.form['modelAnswerText']

    # Access uploaded file
    uploaded_file = request.files.get("studentanswerpdf")

    # Access uploaded file
    uploaded_file2 = request.files.get("modelanswerpdf")


    if uploaded_file is None:
        return "No studentanswerpdf uploaded!", 400  # Handle missing file error

    if uploaded_file2 is None:
        return "No modelanswerpdf uploaded!", 400  # Handle missing file error


    # Get temporary file path
    pdf_path = uploaded_file.filename  # Use the filename for now
    pdf_path2 = uploaded_file2.filename  # Use the filename for now



    # Process the PDF using your Pdf2Image function
    images = Pdf2Image(inputtype=0, path=pdf_path)
    images2 = Pdf2Image(inputtype=0, path=pdf_path2)
    numpages_stuedntans = len(images)
    numpages_modelans =len(images2)
    image_cv=[Pil2cv_converter(image) for image in images]
    image_cv2=[Pil2cv_converter(image) for image in images2]

            
    
    answers= [computer_vision_scanned_version(img, countorss, heightss, widthss) for img in image_cv]
    answers2= [computer_vision_scanned_version(img, countorss, heightss, widthss) for img in image_cv2]

    
    student_answers = [httr_prediction(Httr_model, processor, answer) for answer in answers]
    model_answers = [httr_prediction(Httr_model, processor, answer) for answer in answers2]


    if model_answer_text:
        grades = [pridector(Asag_model, student_answer, model_answer_text, 'easy') for student_answer in student_answers]
    else:
        grades = [pridector(Asag_model, student_answer, model_answer, 'easy') for student_answer, model_answer in zip(student_answers, model_answers)]


 

    if model_answer_text:
        individual_model_answer = model_answer_text
    else:
        for model_ans in model_answers:
            individual_model_answer = model_ans
    
    



    for student_ans in student_answers:
        individual_student_answer = student_ans

    for grade in grades:
        individual_grade = grade
 

  
  
    bothanswers_result = bothanswers_function(individual_student_answer,individual_model_answer,individual_grade,numpages_stuedntans)

    df = pd.DataFrame({'student_answers': student_answers, 'model_answers': model_answers, 'grades': grades})
    df.to_csv('student1.csv', index=False) 

  

    return bothanswers_result


def bothanswers_function(individual_student_answer ,individual_model_answer ,individual_grade,numpages_stuedntans):
    return render_template("New Text Document.html", dispstudentans = individual_student_answer ,dispmodelans = individual_model_answer ,dispgrade = individual_grade ,numquestions = numpages_stuedntans)


from flask import Flask, send_file


@flask_app.route('/download')
def download_file():
    file_path = 'student1.csv'
    return send_file(file_path, as_attachment=True)

