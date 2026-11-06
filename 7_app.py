import os
import sys
import json
import joblib
import numpy as np
import torch
from flask import Flask, render_template, request, session
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# CUSTOM CLASS DEFINITIONS FOR UNPICKLING
# ==========================================
class ScaledModelWrapper(BaseEstimator, RegressorMixin):
    """
    Custom wrapper used when unpickling target-scaled sklearn models.
    Inverse-transforms scaled target predictions back to original values ($).
    """
    def __init__(self, model=None, mean=0.0, std=1.0):
        self.model = model
        self.mean = float(mean)
        self.std = float(std)

    def fit(self, X, y):
        return self

    def predict(self, X):
        if hasattr(self, 'model') and self.model is not None:
            preds = self.model.predict(X)
            return preds * self.std + self.mean
        return np.zeros(len(X))

class TargetScaledRegressor(BaseEstimator, RegressorMixin):
    """
    Legacy target scaling regressor wrapper for backwards compatibility.
    """
    def __init__(self, regressor=None, scaler=None):
        self.regressor = regressor
        self.scaler = scaler

    def fit(self, X, y):
        return self

    def predict(self, X):
        if hasattr(self, 'regressor') and self.regressor is not None:
            preds = self.regressor.predict(X)
            if hasattr(self, 'scaler') and self.scaler is not None and hasattr(self.scaler, 'inverse_transform'):
                preds = preds.reshape(-1, 1) if preds.ndim == 1 else preds
                preds = self.scaler.inverse_transform(preds).ravel()
            return preds
        return np.zeros(len(X))

# Register both wrappers in __main__ module for joblib/pickle lookup compatibility
sys.modules['__main__'].ScaledModelWrapper = ScaledModelWrapper
sys.modules['__main__'].TargetScaledRegressor = TargetScaledRegressor

# ==========================================
# CONFIGURATION SWITCH & DEFAULTS
# ==========================================
# Default fallback mode:
# Set to True  --> Default to multi-model setup (model_apartment/, model_condo/, etc.)
# Set to False --> Default to single unified model setup (/models)
DEFAULT_USE_MULTI_MODEL = False

def calculate_haversine_distance(lat, lon, center_lat=43.651070, center_lon=-79.347015):
    """Calculates geodesic distance (in kilometers) between coordinates and city center."""
    R = 6371.0
    dlat = np.radians(lat - center_lat)
    dlon = np.radians(lon - center_lon)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(center_lat)) * np.cos(np.radians(lat)) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(R * c)

def apply_feature_weights(X_scaled, feature_names):
    """Applies distance feature weights for k-NN inference."""
    weights = np.ones(X_scaled.shape[1], dtype=np.float32)
    for i, name in enumerate(feature_names):
        if name.startswith('unit_'):
            weights[i] = 3.5
        elif name in ['areainfeet', 'bedrooms_num', 'bathrooms_num']:
            weights[i] = 2.5
        elif name in ['lat', 'lon', 'dist_to_center_km', 'parking_spots_num', 'walkscore', 'transitscore', 'bikescore']:
            weights[i] = 2.0
        elif name == 'mls_number_bin':
            weights[i] = 0.5
        elif name in ['pets_no', 'pets_limited', 'pets_yes', 'smoking_no', 'smoking_outdoors', 'smoking_yes']:
            weights[i] = 1.0
        elif name.endswith('_bin'):
            weights[i] = 0.3
    return X_scaled * weights

def parse_binary(val, default=0):
    """Parses form inputs to binary integer (0 or 1)."""
    if val is None:
        return default
    return 1 if str(val).strip().lower() in ['1', 'true', 'yes', 'on'] else 0

def clean_unit_key(unit_name):
    """Standardizes unit type strings to folder suffixes (e.g. 'Duplex/Triplex' -> 'duplex_triplex')."""
    return unit_name.lower().replace('/', '_').replace(' ', '_')

app = Flask(__name__)
app.secret_key = 'rental_valuation_predictor_secret_key'

DEFAULT_UNIT_TYPES = ['Apartment', 'Basement', 'Condo', 'Duplex/Triplex', 'House', 'Townhouse']

# Global container holding both unified and multi-model artifacts
LOADED_DATA = {
    'unified_data': {},
    'multi_data': {}
}

def load_models_from_dir(folder_path, meta):
    """Helper to load PyTorch and Sklearn models defined in models_map."""
    models = {}
    models_map = meta.get('models_map', {})
    for model_display_name, filename in models_map.items():
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            if filename.endswith('.pt'):
                models[model_display_name] = torch.jit.load(file_path)
            else:
                models[model_display_name] = joblib.load(file_path)
    return models

def load_application_artifacts():
    """Loads scalers, metadata, and models for BOTH single unified setup (/models) and multi-model setup (model_*/)."""
    global LOADED_DATA

    # 1. Load Single Unified Model Artifacts (from /models)
    meta_path = 'models/feature_meta.json'
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        models = load_models_from_dir('models', meta)

        unit_cols = [c for c in meta.get('feature_names', []) if c.startswith('unit_')]
        unit_options = [c.replace('unit_', '').capitalize() for c in unit_cols] if unit_cols else DEFAULT_UNIT_TYPES

        LOADED_DATA['unified_data'] = {
            'meta': meta,
            'scaler': scaler,
            'models': models,
            'unit_options': unit_options,
            'model_options': sorted(list(models.keys()))
        }
        print("[App Setup] Unified model loaded from 'models/'")

    # 2. Load Multi-Model Artifacts (from model_apartment/, model_condo/, etc.)
    multi_data = {}
    available_units = []
    all_model_names = set()

    for unit in DEFAULT_UNIT_TYPES:
        key = clean_unit_key(unit)
        folder_path = f"model_{key}"
        meta_p = os.path.join(folder_path, 'feature_meta.json')
        scaler_p = os.path.join(folder_path, 'scaler.pkl')

        if os.path.exists(meta_p):
            with open(meta_p, 'r') as f:
                meta = json.load(f)
            scaler = joblib.load(scaler_p) if os.path.exists(scaler_p) else None
            models = load_models_from_dir(folder_path, meta)

            multi_data[key] = {
                'meta': meta,
                'scaler': scaler,
                'models': models
            }
            available_units.append(unit)
            all_model_names.update(models.keys())

    LOADED_DATA['multi_data'] = {
        'units': multi_data,
        'unit_options': available_units if available_units else DEFAULT_UNIT_TYPES,
        'model_options': sorted(list(all_model_names))
    }
    print(f"[App Setup] Multi-models loaded from 'model_*/' ({len(available_units)} unit types)")

# Initialize artifact loading after custom class registration
load_application_artifacts()

def get_mode_data(use_multi):
    """Retrieves metadata, unit options, and model options for the active mode."""
    if use_multi:
        d = LOADED_DATA['multi_data']
        return d.get('unit_options', DEFAULT_UNIT_TYPES), d.get('model_options', [])
    else:
        d = LOADED_DATA['unified_data']
        return d.get('unit_options', DEFAULT_UNIT_TYPES), d.get('model_options', [])

@app.route('/', methods=['GET'])
def index():
    use_multi = parse_binary(request.args.get('use_multi_model', 1 if DEFAULT_USE_MULTI_MODEL else 0))
    unit_options, model_options = get_mode_data(use_multi)
    
    # Retrieve cached form inputs from session memory and merge with URL query parameters
    cached_form = session.get('form_data', {})
    query_params = request.args.to_dict()
    
    form_data = {**cached_form, **query_params}
    form_data['use_multi_model'] = str(use_multi)
    session['form_data'] = form_data
    
    selected_unit = form_data.get('unittype', unit_options[0] if unit_options else 'Apartment')
    
    selected_model_name = form_data.get('model_choice', '')
    if selected_model_name not in model_options and model_options:
        selected_model_name = model_options[0]

    return render_template(
        'index.html',
        form_data=form_data,
        unit_options=unit_options,
        model_options=model_options,
        selected_unit=selected_unit,
        selected_model=selected_model_name,
        use_multi_model=use_multi
    )

@app.route('/predict', methods=['POST'])
def predict():
    use_multi = parse_binary(request.form.get('use_multi_model', 0))
    try:
        data = request.form.to_dict()
        unit_options, model_options = get_mode_data(use_multi)

        session['form_data'] = data

        # 1. Parse dropdown inputs
        selected_unit = data.get('unittype', unit_options[0] if unit_options else 'Apartment')
        selected_model_name = data.get('model_choice', model_options[0] if model_options else '')

        # 2. Retrieve appropriate Scaler, Model, and Metadata based on active mode
        if use_multi:
            unit_key = clean_unit_key(selected_unit)
            unit_artifact = LOADED_DATA['multi_data']['units'].get(unit_key)
            if not unit_artifact:
                raise ValueError(f"No trained multi-model found for unit type '{selected_unit}'.")
            meta = unit_artifact['meta']
            scaler = unit_artifact['scaler']
            loaded_models = unit_artifact['models']
        else:
            meta = LOADED_DATA['unified_data']['meta']
            scaler = LOADED_DATA['unified_data']['scaler']
            loaded_models = LOADED_DATA['unified_data']['models']

        if selected_model_name not in loaded_models:
            raise ValueError(f"Selected model '{selected_model_name}' is not available in current setup mode.")

        target_model = loaded_models[selected_model_name]
        feature_names = meta.get('feature_names', [])

        # 3. Core numeric inputs
        listed_price = float(data.get('price', 0))
        area = float(data.get('areainfeet', meta.get('median_area', 800)))
        bedrooms = float(data.get('bedrooms_num', 1))
        bathrooms = float(data.get('bathrooms_num', 1))
        parking_spots = float(data.get('parking_spots_num', 0))
        lat = float(data.get('lat', 43.6532))
        lon = float(data.get('lon', -79.3832))

        dist_to_center_km = calculate_haversine_distance(lat, lon)

        # 4. Policy & Lease terms
        pets_selection = data.get('pets_allowed_policy', 'no').lower().strip()
        pets_no = 1 if pets_selection == 'no' else 0
        pets_limited = 1 if pets_selection == 'limited' else 0
        pets_yes = 1 if pets_selection == 'yes' else 0

        smoking_selection = data.get('smoking_policy', 'no').lower().strip()
        smoking_no = 1 if smoking_selection == 'no' else 0
        smoking_outdoors = 1 if smoking_selection == 'outdoors' else 0
        smoking_yes = 1 if smoking_selection == 'yes' else 0

        binary_inputs = {
            'pets_no': pets_no,
            'pets_limited': pets_limited,
            'pets_yes': pets_yes,
            'smoking_no': smoking_no,
            'smoking_outdoors': smoking_outdoors,
            'smoking_yes': smoking_yes,
            'airconditioning_bin': parse_binary(data.get('airconditioning_bin', 0)),
            'laundryinunit_bin': parse_binary(data.get('laundryinunit_bin', 0)),
            'dishwasher_bin': parse_binary(data.get('dishwasher_bin', 0)),
            'furnished_bin': parse_binary(data.get('furnished_bin', 0)),
            'gym_bin': parse_binary(data.get('gym_bin', 0)),
            'balcony_bin': parse_binary(data.get('balcony_bin', 0)),
            'yard_bin': parse_binary(data.get('yard_bin', 0)),
            'hydro_bin': parse_binary(data.get('hydro_bin', 0)),
            'heat_bin': parse_binary(data.get('heat_bin', 0)),
            'water_bin': parse_binary(data.get('water_bin', 0)),
            'agreement_1yr_bin': parse_binary(data.get('agreement_1yr_bin', 0)),
            'mls_number_bin': parse_binary(data.get('mls_number_bin', 0)),
        }

        # 5. One-Hot Encoding for Unit Type
        unit_values = {}
        for col in feature_names:
            if col.startswith('unit_'):
                col_unit_name = col.replace('unit_', '').lower().replace('/', '_').replace(' ', '')
                sel_clean = clean_unit_key(selected_unit)
                unit_values[col] = 1 if col_unit_name == sel_clean else 0

        input_dict = {
            'Price($)': listed_price,
            'areainfeet': area,
            'bedrooms_num': bedrooms,
            'bathrooms_num': bathrooms,
            'parking_spots_num': parking_spots,
            'dist_to_center_km': dist_to_center_km,
            'lat': lat,
            'lon': lon,
            **binary_inputs,
            **unit_values
        }

        feature_vector = np.array([[input_dict.get(col, 0.0) for col in feature_names]], dtype=np.float32)

        # 6. Feature scaling & inference
        scaled_input = scaler.transform(feature_vector) if scaler else feature_vector

        if isinstance(target_model, (torch.jit.ScriptModule, torch.nn.Module)):
            target_model.eval()
            with torch.no_grad():
                tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
                raw_pred = target_model(tensor_input).detach().cpu().numpy()
        else:
            # Apply feature weighting specifically for k-NN distance metrics
            if 'knn' in selected_model_name.lower() or 'k-nn' in selected_model_name.lower():
                model_input = apply_feature_weights(scaled_input, feature_names)
            else:
                model_input = scaled_input
            raw_pred = target_model.predict(model_input)

        predicted_fair_value = float(np.ravel(raw_pred)[0])

        # 7. Valuation analysis
        residual = listed_price - predicted_fair_value
        magnitude = abs(residual)
        tolerance = 0.08 * predicted_fair_value

        if residual > tolerance:
            status = "Overvalued"
            color_class = "danger"
            summary = f"This property is listed ${magnitude:,.2f} ABOVE predicted fair market value."
        elif residual < -tolerance:
            status = "Undervalued"
            color_class = "success"
            summary = f"This property is listed ${magnitude:,.2f} BELOW predicted fair market value."
        else:
            status = "Fairly Valued"
            color_class = "info"
            summary = f"This property is listed within reasonable market expectation (±${tolerance:,.2f})."

        result = {
            'selected_unit': selected_unit,
            'selected_model': selected_model_name,
            'listed_price': listed_price,
            'predicted_fair_value': predicted_fair_value,
            'status': status,
            'magnitude': magnitude,
            'summary': summary,
            'color_class': color_class,
            'mode_label': "Multi-Model Setup" if use_multi else "Single Unified Model (/models)"
        }

        return render_template(
            'index.html',
            result=result,
            form_data=data,
            unit_options=unit_options,
            model_options=model_options,
            selected_unit=selected_unit,
            selected_model=selected_model_name,
            use_multi_model=use_multi
        )

    except Exception as e:
        unit_options, model_options = get_mode_data(use_multi)
        data = request.form.to_dict()
        session['form_data'] = data
        selected_unit = data.get('unittype', unit_options[0] if unit_options else 'Apartment')
        selected_model_name = data.get('model_choice', model_options[0] if model_options else '')
        return render_template(
            'index.html',
            error=str(e),
            form_data=data,
            unit_options=unit_options,
            model_options=model_options,
            selected_unit=selected_unit,
            selected_model=selected_model_name,
            use_multi_model=use_multi
        )

if __name__ == '__main__':
    app.run(debug=True, port=5050)