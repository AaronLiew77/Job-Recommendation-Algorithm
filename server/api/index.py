from flask import Flask, jsonify, request
from flask_cors import CORS
from server import create_app

app = create_app()
CORS(app)

def handler(request):
    return app(request)