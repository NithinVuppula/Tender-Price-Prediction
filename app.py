from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'static/model.pkl'
pipeline = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('data.html', message="Error: No file part")

        file = request.files['file']
        
        if file.filename == '':
            return render_template('data.html', message="Error: No selected file")

        if file and file.filename.endswith('.csv'):
            # Read the uploaded CSV file into a DataFrame
            df = pd.read_csv(file)

            # Check if required columns are present in the file
            required_columns = ['qty', 'basic', 'Comp_A', 'Comp_B', 'Comp_C', 'Comp_D', 'Comp_E', 'Comp_F', 'Comp_G', 'Comp_H', 'Comp_I', 'Comp_J', 'Comp_K', 'Comp_L']
            
            if not all(col in df.columns for col in required_columns):
                return render_template('data.html', message="Error: Missing required columns in the input CSV")
            
            # Prepare the features for prediction
            features = df[required_columns]

            # Predict using the trained model
            predictions = pipeline.predict(features)

            # Add predictions to the DataFrame
            df['predicted_l1_price'] = predictions

            # Render the result in a table format on the webpage
            prediction_data = df.to_html(classes='table table-hover table-bordered table-striped', index=False)

            return render_template('data.html', prediction_data=prediction_data, message=None)

        else:
            return render_template('data.html', message="Error: Invalid file format. Please upload a CSV file.")

    except Exception as e:
        return render_template('data.html', message=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=False)