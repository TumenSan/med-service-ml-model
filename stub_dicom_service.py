from flask import Flask, request, jsonify
from datetime import datetime
import time

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process_dicom():
    data = request.json
    print("Stub получил данные:", data)
    time.sleep(1)  # Имитация задержки

    response = {
        "task_id": data.get("taskId"),
        "model_id": data.get("modelId"),
        "data": data.get("parameters"),
        #"data": "{\"bloodPressure\": \"120/80\",\"glucose\": 5.4}",
        "conclusion": "stub_pneumonia"
        #"timestamp": datetime.now().isoformat(),
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
