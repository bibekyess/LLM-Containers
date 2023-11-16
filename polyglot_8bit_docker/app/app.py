import json
from flask import Flask, jsonify, g
from flask_cors import CORS
from flask_expects_json import expects_json
from werkzeug.exceptions import HTTPException, UnprocessableEntity

from helpers.inference import PolyglotGenerator

# The flask api for serving predictions
app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
CORS(app)

# Expected JSON Schema of the predict request
with open("schema.json", "r") as f:
    SCHEMA = json.loads(f.read())


@app.errorhandler(HTTPException)
def handle_exception(e):
    """
    To keep all responses consistently JSON.
    Return JSON instead of HTML for HTTP errors.
    """
    return (
        jsonify(
            {
                "code": e.code,
                "name": e.name,
                "description": e.description,
            }
        ),
        e.code,
    )


@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
@app.route("/health_check", methods=["GET"])
def health_check():
    """
    The healh check makes sure container is ok.
    For example, check the model, database connections (if present), etc.
    """
    # Warm-up the model with health check
    PolyglotGenerator.load()
    return jsonify({"success": True}), 200


@app.route(
    "/generate",
    methods=["POST"],
)
@expects_json(SCHEMA)
def generate():
    """
    The main predict endpoint.
    """
    # Payload is checked for the right schema using flask_expects_json
    # If payload is invalid, request will be aborted with error code 400
    # If payload is valid it is stored in g.data
    request_obj = g.data
    query = request_obj["text"]

    # Get the prediction.
    prediction = PolyglotGenerator.generate_text(prompt=query)

    # Don't return as json object as it will encode korean script
    return prediction

if __name__ == "__main__":
    # Do not set debug=True in production
    app.run(host="0.0.0.0", port=5000, debug=True)
