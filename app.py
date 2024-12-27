from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Update these paths to your local Windows paths
model_path = "H:\My Drive\HateSpeechDetection\BerTweet/BerTweet_trained_model"
tokenizer_path = "H:\My Drive\HateSpeechDetection\BerTweet/BerTweet_trained_tokenizer"

# Create classifier pipeline
classifier = pipeline("text-classification", 
                     model=model_path, 
                     tokenizer=tokenizer_path,
                     local_files_only=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']
            prediction = classifier(text)[0]
            
            # Debug print
            print("Raw prediction:", prediction)
            
            # Check the actual label value
            print("Label type:", type(prediction['label']))
            print("Label value:", prediction['label'])
            
            # Convert numerical labels to readable text
            # Adjust these conditions based on your model's output
            if prediction['label'] == 'LABEL_1':
                label = "Hate Speech"
                label_color = "#dc3545"
            else:
                label = "Non-Hate Speech"
                label_color = "#28a745"
            
            return jsonify({
                'prediction': label,
                'confidence': prediction['score'],
                'color': label_color
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)