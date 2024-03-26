# import library dependencies, tested on python 3.9.6
import gradio as gr
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# load the pre-trained model from the appropriate file path
def predict_plant(path):
    model = tf.keras.models.load_model('saved_model/protein_model.keras')

    # redefine values from the model
    img_height = img_width = 180
    class_names = ['2fak', 'coagulation_factor_v']
    
    # load the image into a variable
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )

    # convert the image into a tensor and create a batch for testing
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # find the confidence probability for each plant
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    confidences = {class_names[i]: float(score[i]) for i in range(len(class_names))}
    return confidences

# add a title and description to the model
title = "Protein Picker Model"
description = """
              Protein Picker Model. A CS562 Project.
              """

# launch the app
app = gr.Interface(
    fn=predict_plant,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Label(num_top_classes=2),
    flagging_options=["incorrect", "other"],
    title=title,
    description=description,
    examples=[
        os.path.join(os.path.dirname(__file__), "coagulation_test.png"),
        os.path.join(os.path.dirname(__file__), "2FAK_test.png"),
    ],
)
app.launch(share=True)