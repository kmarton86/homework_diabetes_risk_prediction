from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    data = request.form

    # Sex konverzió
    sex_map = {
        "female": 0,
        "male": 1
    }

    new_patient = {
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

    print("Received patient:", new_patient)

    return jsonify(new_patient)


if __name__ == '__main__':
    app.run(debug=True)