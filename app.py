from flask import Flask, jsonify, request
from flask_cors import CORS
from neural_network import NeuralNetwork
from nn import procesar_data


from preprocesamiento_data import sacar_espectrograma, normalizar, binarizar, obtener_dataframe

app = Flask(__name__)

CORS(app)

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        'message': 'Hello from the server!'
    }
    return jsonify(data)

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    # Verifica que se haya enviado un archivo de audio
    if 'audio' not in request.files:
        return jsonify({'error': 'No se encontró un archivo de audio'}), 400

    audio_file = request.files['audio']

    # Verifica que se haya enviado un archivo de audio válido
    if audio_file.filename == '':
        return jsonify({'error': 'No se seleccionó un archivo de audio válido'}), 400
    
    if 'etiqueta' not in request.form:
        return jsonify({'error': 'No se proporcionó una etiqueta'}), 400
    
    etiqueta = request.form['etiqueta']

    try:
        print('Antes del path')
        audio_path = './audios/' + audio_file.filename
        print('Despues del path')
        audio_file.save(audio_path)
        cancion = sacar_espectrograma(audio_path, etiqueta)
        print('Despues de sacar espectrograma')
        df_normalizado = obtener_dataframe(normalizar(cancion))
        print('Despues de normalizar')
        df_unido = obtener_dataframe(binarizar(df_normalizado,0.5))
        print(df_unido)
        df_unido.to_csv('./data/cancion.csv', index=False)

        porcentaje = procesar_data('./data/cancion.csv')
        return jsonify(porcentaje), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)