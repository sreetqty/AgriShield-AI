import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from urllib.parse import quote

# -------------------------------
# 1. Load trained model
# -------------------------------

MODEL_PATH = "AgriShield_AI_Final.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf",
    "Bell_pepper leaf spot",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "grape leaf",
    "grape leaf black rot"
]

# -------------------------------
# 2. Management information
# -------------------------------

disease_info = {
    "Tomato Septoria leaf spot": (
        "Remove heavily affected lower leaves. Avoid overhead irrigation. "
        "Maintain good airflow and sanitation. Consult a local agricultural "
        "expert regarding appropriate fungicide options."
    ),

    "Tomato Early blight leaf": (
        "Remove infected leaves where practical. Avoid wetting foliage during "
        "irrigation. Rotate crops and maintain plant spacing. Seek local "
        "guidance for suitable fungicide use."
    ),

    "Tomato leaf late blight": (
        "Remove severely infected plant material carefully. Avoid overhead "
        "irrigation and monitor nearby plants. Late blight can spread rapidly; "
        "consult an agricultural expert promptly."
    ),

    "Potato leaf early blight": (
        "Remove affected foliage when practical, avoid overhead watering, "
        "maintain crop hygiene and rotate crops. Consult local experts "
        "regarding fungicide options."
    ),

    "Potato leaf late blight": (
        "Inspect the crop immediately and remove severely affected plants "
        "where appropriate. Avoid overhead irrigation and seek prompt "
        "agricultural advice."
    ),

    "Squash Powdery mildew leaf": (
        "Improve airflow, avoid excessive nitrogen, remove severely affected "
        "leaves and monitor the crop. Consult an agricultural expert about "
        "appropriate disease management."
    )
}

# -------------------------------
# 3. Management lookup
# -------------------------------

def get_management_guidance(disease):

    if disease in disease_info:
        return disease_info[disease]

    return (
        "Specific management guidance is not yet available in the prototype "
        "database for this predicted condition. Please consult a local "
        "agricultural expert before applying treatment."
    )

# -------------------------------
# 4. Prediction function
# -------------------------------

def predict_leaf(image):

    if image is None:
        return (
            "Please upload or capture a leaf image.",
            "",
            "",
            "",
            "The SMS link will appear after a prediction."
        )

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    probabilities = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    predicted_index = top_indices[0]
    predicted_disease = class_names[predicted_index]
    confidence = float(probabilities[predicted_index] * 100)

    # Confidence-based status
    if confidence >= 70:
        status = "Prediction confidence accepted for prototype display"
        prediction_text = (
            f"Predicted condition: {predicted_disease}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"Status: Prediction above prototype threshold"
        )
    else:
        status = "Uncertain prediction"
        prediction_text = (
            f"Predicted condition: {predicted_disease}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"Status: Uncertain prediction"
        )

    # Top 3 predictions
    top3_text = ""

    for rank, index in enumerate(top_indices, start=1):

        top3_text += (
            f"{rank}. {class_names[index]}: "
            f"{probabilities[index] * 100:.2f}%\n"
        )

    # Management guidance
    guidance = get_management_guidance(predicted_disease)

    # Farmer alert
    alert_message = (
        "AgriShield AI Crop Alert\n"
        f"Predicted condition: {predicted_disease}\n"
        f"Model confidence: {confidence:.2f}%\n"
        f"Status: {status}\n\n"
        "This is an AI-assisted prediction, not a confirmed diagnosis. "
        "Please verify with an agricultural expert."
    )

    sms_link = "sms:?body=" + quote(alert_message)

    sms_markdown = (
        f"[📩 Open SMS to send alert]({sms_link})"
    )

    return (
        prediction_text,
        top3_text,
        guidance,
        alert_message,
        sms_markdown
    )

# -------------------------------
# 5. Gradio interface
# -------------------------------

with gr.Blocks(title="AgriShield AI") as agrishield_app:

    gr.Markdown("""
    # 🌿 AgriShield AI
    ### Smart Crop Disease Detection and Farmer Alert System

    Upload or capture a crop leaf image to receive an AI-assisted
    disease prediction and management guidance.
    """)

    image_input = gr.Image(
        label="Capture or Upload Leaf Image",
        type="pil",
        sources=["upload", "webcam"]
    )

    predict_button = gr.Button(
        "🔍 Analyze Leaf",
        variant="primary"
    )

    prediction_output = gr.Textbox(
        label="AI Prediction",
        lines=4
    )

    top3_output = gr.Textbox(
        label="Top 3 Predictions",
        lines=4
    )

    guidance_output = gr.Textbox(
        label="Management Guidance",
        lines=5
    )

    alert_output = gr.Textbox(
        label="Farmer Alert Message",
        lines=6
    )

    sms_output = gr.Markdown(
        "The SMS link will appear after prediction."
    )

    predict_button.click(
        fn=predict_leaf,
        inputs=image_input,
        outputs=[
            prediction_output,
            top3_output,
            guidance_output,
            alert_output,
            sms_output
        ]
    )

    gr.Markdown("""
    ---
    **Disclaimer:** AgriShield AI provides AI-assisted predictions.
    Predictions are not confirmed diagnoses. Please verify uncertain
    results with agricultural experts.
    """)

# -------------------------------
# 6. Cloud launch
# -------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    agrishield_app.launch(
        server_name="0.0.0.0",
        server_port=port
    )
