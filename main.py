from flask import Flask, jsonify, request
from abecedario.abecedario_api import abecedario_api
from palabras.palabras_api import palabras_api

app = Flask(__name__)

# Registrar blueprints
app.register_blueprint(abecedario_api)
app.register_blueprint(palabras_api)

# Ruta de prueba
@app.route('/')
def index():
    return jsonify({'message': 'Bienvenido al API de lenguaje de señas en tiempo real'})

if __name__ == '__main__':
    app.run()
