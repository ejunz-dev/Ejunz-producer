from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from app.models.question import Question
from app.mcq_generation import MCQGenerator
from trainer import ModelTrainer
import os

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

trainer = ModelTrainer()
MQC_Generator = MCQGenerator()
fine_tuned_model = None


@app.route("/")
@cross_origin()
def hello():
    return jsonify(message="Generator test")


@app.route("/generate", methods=["POST"])
@cross_origin()
def generate():
    try:
        request_json = request.get_json()
        if not request_json or 'text' not in request_json:
            return jsonify({"status": "error", "message": "Text is required"}), 400

        text = request_json['text'].strip()
        count = int(request_json.get('count', 10))

        if not text:
            return jsonify({"status": "error", "message": "Text cannot be empty"}), 400
        if count <= 0 or count > 100:
            return jsonify({"status": "error", "message": "Count must be a positive integer between 1 and 100"}), 400

        questions = MQC_Generator.generate_mcq_questions(text, count)
        result = [{"question": q.questionText, "answers": [q.answerText] + q.distractors} for q in questions]

        return jsonify(result=result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/fine_tune", methods=["POST"])
@cross_origin()
def fine_tune():
    try:
        request_json = request.get_json()

        model_name = request_json.get("model_name")
        dataset = request_json.get("dataset")
        epochs = request_json.get("epochs", 3)

        if not model_name or not dataset:
            return jsonify({"status": "error", "message": "model_name and dataset are required"}), 400

        result = trainer.fine_tune_model(model_name, dataset, epochs)
        return jsonify({"status": "success", "message": "Model fine-tuned successfully", "result": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/load_model", methods=["POST"])
@cross_origin()
def load_model():
    try:
        request_json = request.get_json()
        model_path = request_json.get("model_path")

        if not model_path:
            return jsonify({"status": "error", "message": "model_path is required"}), 400

        global fine_tuned_model
        fine_tuned_model = trainer.load_model(model_path)

        return jsonify({"status": "success", "message": "Model loaded successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/generate_with_model", methods=["POST"])
@cross_origin()
def generate_with_model():
    try:
        global fine_tuned_model
        if fine_tuned_model is None:
            return jsonify({"status": "error", "message": "Fine-tuned model is not loaded"}), 400

        request_json = request.get_json()
        text = request_json.get("text")
        count = int(request_json.get("count", 10))

        if not text:
            return jsonify({"status": "error", "message": "Text is required"}), 400

        tokenizer = fine_tuned_model["tokenizer"]
        model = fine_tuned_model["model"]

        inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
        outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        num_return_sequences=count,
        num_beams=5,
        temperature=1.2,
        top_k=50,
        top_p=0.9,
        early_stopping=True
    )

        result = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9002)
