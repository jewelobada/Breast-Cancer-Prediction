import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model import BreastCancerModel

# Unique instance name
tumor_diagnostic_app = Flask(__name__)

# Renamed model instance
clinical_predictor = BreastCancerModel()

def initialize_diagnostic_engine():
    """Validates model availability and handles auto-training if needed"""
    if not clinical_predictor.load_model():
        print("Pre-trained weights not found. Starting fresh training session...")
        from model import train_and_save_model
        train_and_save_model()
        clinical_predictor.load_model()

# Execute setup
initialize_diagnostic_engine()

@tumor_diagnostic_app.route('/')
def index_view():
    """Serves the primary diagnostic interface"""
    return render_template('index.html', feature_names=clinical_predictor.feature_names)

@tumor_diagnostic_app.route('/predict', methods=['POST'])
def process_biopsy_data():
    """Evaluates biopsy metrics to determine tumor classification"""
    try:
        biopsy_metrics = request.get_json()
        validated_input = {}

        # Integrity Check: ensure all required cell characteristics are present
        for metric_name in clinical_predictor.feature_names:
            if metric_name not in biopsy_metrics:
                return jsonify({
                    'success': False,
                    'error': f'Metric missing: {metric_name}'
                })

            try:
                numeric_value = float(biopsy_metrics[metric_name])
                if numeric_value < 0:
                    return jsonify({
                        'success': False,
                        'error': f'Field {metric_name} must contain a non-negative value'
                    })
                validated_input[metric_name] = numeric_value
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Invalid numeric entry for: {metric_name}'
                })

        # Generate Inference
        class_label, prob_score = clinical_predictor.predict(validated_input)

        # Map labels (0: Malignant, 1: Benign)
        is_non_cancerous = (class_label == 1)
        certainty_level = prob_score if is_non_cancerous else (1.0 - prob_score)

        return jsonify({
            'success': True,
            'diagnosis': 'Benign' if is_non_cancerous else 'Malignant',
            'is_benign': bool(is_non_cancerous),
            'confidence': float(round(certainty_level * 100, 1)),
            'message': '✓ Analysis suggests a BENIGN (non-cancerous) tumor' if is_non_cancerous
                       else '⚠️ Analysis suggests a MALIGNANT (cancerous) tumor',
            'disclaimer': 'This AI tool is for academic use only. Consult medical professionals for clinical validation.'
        })

    except Exception as failure:
        return jsonify({
            'success': False,
            'error': str(failure)
        })

@tumor_diagnostic_app.route('/sample-data')
def fetch_example_metrics():
    """Retrieves comparative samples from the dataset for UI testing"""
    try:
        data_directory = os.path.dirname(os.path.abspath(__file__))
        csv_source = os.path.join(data_directory, 'data', 'breast_cancer.csv')

        raw_df = pd.read_csv(csv_source)

        # Extract representative cases (Class 1 = Benign, Class 0 = Malignant)
        case_benign = raw_df[raw_df['diagnosis'] == 1].iloc[0].drop('diagnosis').to_dict()
        case_malignant = raw_df[raw_df['diagnosis'] == 0].iloc[0].drop('diagnosis').to_dict()

        return jsonify({
            'success': True,
            'benign_sample': case_benign,
            'malignant_sample': case_malignant
        })
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': str(exc)
        })

if __name__ == '__main__':
    # Local dev server initialization
    tumor_diagnostic_app.run(debug=True, host='0.0.0.0', port=5000)