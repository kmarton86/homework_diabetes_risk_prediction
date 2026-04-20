from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


# ===== REST API ENDPOINT =====
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    sex_map = {
        "female": 0,
        "male": 1
    }

    try:
        patient = {
            "age": int(data.get("age")),
            "sex": sex_map.get(data.get("sex")),
            "bmi": float(data.get("bmi")),
            "bp": int(data.get("bp")),
            "s1": int(data.get("s1")),
            "s2": int(data.get("s2")),
            "s3": int(data.get("s3")),
            "s4": int(data.get("s4")),
            "s5": float(data.get("s5")),
            "s6": int(data.get("s6"))
        }
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    print("Received patient:", patient)

    # IDE JÖN MAJD A MODEL (később)
    fake_prediction = "Not Endangered"

    return jsonify({
        "patient": patient,
        "prediction": fake_prediction
    })


if __name__ == '__main__':
    app.run(debug=True)