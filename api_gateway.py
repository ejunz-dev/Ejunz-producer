from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
from app.models.question import Question
from app.mcq_generation import MCQGenerator

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

MQC_Generator = MCQGenerator()

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
        count = 10 if 'count' not in request_json or request_json['count'] == '' else int(request_json['count'])

        if not text:
            return jsonify({"status": "error", "message": "Text cannot be empty"}), 400
        if count <= 0:
            return jsonify({"status": "error", "message": "Count must be a positive integer"}), 400

        questions = MQC_Generator.generate_mcq_questions(text, count)

        result = [{"question": q.questionText, "answers": [q.answerText] + q.distractors} for q in questions]

        return jsonify(result=result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 9002, app)
