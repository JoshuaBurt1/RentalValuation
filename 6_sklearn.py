import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

np.random.seed(42)

# ==========================================
# SKLEARN INFERENCE WRAPPERS
# ==========================================
class ScaledModelWrapper:
    """Wrapper for Sklearn models to inverse-transform target predictions automatically during inference."""
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = float(mean)
        self.std = float(std)
        
    def predict(self, X):
        return self.model.predict(X) * self.std + self.mean

# ==========================================
# DATA PREPROCESSING & FEATURE ENCODING
# ==========================================
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Normalize column names to lowercase to ensure casing compatibility
    df.columns = df.columns.str.strip().str.lower()
    
    # Identify price column
    price_col = 'price($)' if 'price($)' in df.columns else ('price' if 'price' in df.columns else None)
    if price_col is None:
        raise KeyError("Could not find price column in CSV.")

    # Data is pre-cleaned; preserve all rows without dropping or filtering
    df_clean = df.copy()

    # Numeric features parsing
    median_area = df_clean['areainfeet'].median() if 'areainfeet' in df_clean.columns else 700.0
    df_clean['areainfeet'] = df_clean['areainfeet'].fillna(median_area) if 'areainfeet' in df_clean.columns else median_area

    # Coordinates
    if 'locationcoordinates' in df_clean.columns:
        coords = df_clean['locationcoordinates'].astype(str).str.split(',', expand=True)
        df_clean['lat'] = pd.to_numeric(coords[0], errors='coerce').fillna(43.651070)
        df_clean['lon'] = pd.to_numeric(coords[1], errors='coerce').fillna(-79.347015)
    else:
        df_clean['lat'] = df_clean.get('lat', 43.651070)
        df_clean['lon'] = df_clean.get('lon', -79.347015)

    # Distance to Center Parsing
    def parse_distance(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).replace(',', '.')
        nums = re.findall(r'\d+\.?\d*', val_str)
        return float(nums[0]) if nums else 0.0

    if 'dist_to_center_km' in df_clean.columns and df_clean['dist_to_center_km'].notna().any():
        df_clean['dist_to_center_km'] = df_clean['dist_to_center_km'].apply(parse_distance)
    elif 'citycentredistance' in df_clean.columns:
        df_clean['dist_to_center_km'] = df_clean['citycentredistance'].apply(parse_distance)
    else:
        df_clean['dist_to_center_km'] = 0.0

    # Binary parsing for mls_number_bin (Strict Binary 0.0 / 1.0)
    mls_col = 'mls_number_bin' if 'mls_number_bin' in df_clean.columns else ('mls_number' if 'mls_number' in df_clean.columns else None)
    if mls_col is not None:
        mls_str = df_clean[mls_col].astype(str).str.lower().str.strip()
        is_bool = mls_str.isin(['yes', 'true', '1', 'no', 'false', '0'])
        if is_bool.any():
            df_clean['mls_number_bin'] = mls_str.isin(['yes', 'true', '1']).astype(np.float32)
        else:
            extracted_mls = mls_str.str.extract(r'(\d+)')[0]
            df_clean['mls_number_bin'] = (pd.to_numeric(extracted_mls, errors='coerce').fillna(0) > 0).astype(np.float32)
    else:
        df_clean['mls_number_bin'] = 0.0

    # Continuous numerical score parsing for walkscore, transitscore, bikescore
    for score_col in ['walkscore', 'transitscore', 'bikescore']:
        if score_col in df_clean.columns:
            extracted_score = df_clean[score_col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
            score_series = pd.to_numeric(extracted_score, errors='coerce')
            median_val = score_series.median() if not score_series.dropna().empty else 50.0
            df_clean[score_col] = score_series.fillna(median_val).astype(np.float32)
        else:
            df_clean[score_col] = 50.0

    def parse_bedrooms(val):
        if pd.isna(val): return 1.0
        val = str(val).strip().lower()
        if 'bachelor' in val or 'studio' in val: return 0.5
        nums = re.findall(r'\d+', val)
        if not nums: return 1.0
        base = float(nums[0])
        if 'den' in val: base += 0.5
        return base

    def parse_bathrooms(val):
        if pd.isna(val): return 1.0
        val = str(val).strip()
        nums = re.findall(r'\d+\.?\d*', val)
        return float(nums[0]) if nums else 1.0

    def parse_parking(val):
        if pd.isna(val): return 0.0
        nums = re.findall(r'\d+', str(val))
        return float(nums[0]) if nums else 0.0

    df_clean['bedrooms_num'] = df_clean['numberbedrooms'].apply(parse_bedrooms) if 'numberbedrooms' in df_clean.columns else 1.0
    df_clean['bathrooms_num'] = df_clean['numberbathrooms'].apply(parse_bathrooms) if 'numberbathrooms' in df_clean.columns else 1.0
    df_clean['parking_spots_num'] = df_clean['numberparkingspots'].apply(parse_parking) if 'numberparkingspots' in df_clean.columns else 0.0

    # FULL ONE-HOT ENCODING FOR PETS (3-State Columns)
    if 'petsallowed' in df_clean.columns:
        pet_str = df_clean['petsallowed'].astype(str).str.lower()
        df_clean['pets_no'] = pet_str.isin(['no', 'false', '0']).astype(int)
        df_clean['pets_limited'] = pet_str.str.contains('limit').astype(int)
        df_clean['pets_yes'] = pet_str.isin(['yes', 'true', '1']).astype(int)
    else:
        df_clean['pets_no'], df_clean['pets_limited'], df_clean['pets_yes'] = 0, 0, 0

    # FULL ONE-HOT ENCODING FOR SMOKING (3-State Columns)
    if 'smokingpermitted' in df_clean.columns:
        smoke_str = df_clean['smokingpermitted'].astype(str).str.lower()
        df_clean['smoking_no'] = smoke_str.isin(['no', 'false', '0']).astype(int)
        df_clean['smoking_outdoors'] = smoke_str.str.contains('outdoor|balcony|patio').astype(int)
        df_clean['smoking_yes'] = smoke_str.isin(['yes', 'true', '1']).astype(int)
    else:
        df_clean['smoking_no'], df_clean['smoking_outdoors'], df_clean['smoking_yes'] = 0, 0, 0

    policy_cols = ['pets_no', 'pets_limited', 'pets_yes', 'smoking_no', 'smoking_outdoors', 'smoking_yes']

    # FULL ONE-HOT ENCODING FOR UNIT TYPE
    unit_dummies = pd.DataFrame(index=df_clean.index)
    if 'unittype' in df_clean.columns and df_clean['unittype'].nunique() > 1:
        unit_dummies = pd.get_dummies(df_clean['unittype'], prefix='unit', drop_first=False).astype(int)

    # Standardizing Binary Amenities
    standard_amenity_cols = [
        'heat', 'hydro', 'water', 
        'airconditioning', 'dishwasher', 'fridge-freezer', 'laundryinunit', 'laundryinbuilding', 
        'balcony', 'yard', 'bicycleparking',
        'internet', 'cabletv', 'furnished', 'concierge', 'twentyfourhoursecurity', 'elevator', 'gym', 'pool', 'storagelocker',
        'accessiblewashroomsinsuite', 'audioprompts', 'barrierfreeentrancesandramps', 'braillelabels', 'visualaids', 'wheelchairaccessible'
    ]

    amenity_bin_cols = []
    for col in standard_amenity_cols:
        if col in df_clean.columns:
            clean_col_name = col.replace('-', '') + '_bin'
            df_clean[clean_col_name] = df_clean[col].astype(str).str.lower().isin(['yes', '1', 'true']).astype(int)
            if clean_col_name not in amenity_bin_cols:
                amenity_bin_cols.append(clean_col_name)

    df_clean['agreement_1yr_bin'] = df_clean['agreementtype'].astype(str).str.lower().str.contains('1 year|1yr|1').astype(int) if 'agreementtype' in df_clean.columns else 0

    # Base continuous numerical features
    base_numeric_cols = [
        'areainfeet', 'lat', 'lon', 'dist_to_center_km', 
        'bedrooms_num', 'bathrooms_num', 'parking_spots_num',
        'walkscore', 'transitscore', 'bikescore'
    ]
    
    # Binary indicator features
    binary_cols = ['mls_number_bin'] + amenity_bin_cols + policy_cols + ['agreement_1yr_bin']

    X_df = pd.concat([df_clean[base_numeric_cols + binary_cols], unit_dummies], axis=1)
    
    # Filter non-constant attributes to eliminate zero-variance variables
    non_constant_cols = [col for col in X_df.columns if X_df[col].nunique() > 1]
    X_df = X_df[non_constant_cols]
    
    active_binary_cols = [c for c in binary_cols if c in X_df.columns]
    target_y = df_clean[price_col].values.astype(np.float32)
    
    return (
        X_df, 
        df_clean, 
        target_y, 
        active_binary_cols, 
        unit_dummies.columns.tolist() if not unit_dummies.empty else [], 
        [], 
        median_area
    )

def apply_feature_weights(X_scaled, feature_names):
    """Applies feature scaling weights for KNN distance calculations."""
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

def classify_valuation(listed_price, fair_price, threshold=0.10):
    """Categorizes listings into 0: Undervalued, 1: Fairly Valued, 2: Overvalued."""
    ratio = listed_price / np.maximum(fair_price, 1.0)
    classes = np.ones_like(listed_price, dtype=int)
    classes[ratio < (1.0 - threshold)] = 0
    classes[ratio > (1.0 + threshold)] = 2
    return classes

# ==========================================
# SKLEARN TRAINING & INFERENCE HELPERS
# ==========================================
def train_sklearn_iterative(model_base, X_train, y_train, X_val, y_val, epochs=150, batch_size=128, is_scaled=False):
    """
    Minibatched iterative trainer that mimics the original PyTorch loop,
    allowing for smooth validation history charting and target scaling.
    """
    model = clone(model_base)
    
    train_hist, val_hist = [], []
    num_samples = X_train.shape[0]
    
    # Scale Targets Internally
    if is_scaled:
        y_mean, y_std = y_train.mean(), y_train.std()
        y_tr_fit = (y_train - y_mean) / y_std
        y_va_fit = (y_val - y_mean) / y_std
    else:
        y_mean, y_std = 0.0, 1.0
        y_tr_fit = y_train
        y_va_fit = y_val
        
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        
        # Minibatch iteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                model.partial_fit(X_train[batch_idx], y_tr_fit[batch_idx])
            
        tr_pred = model.predict(X_train)
        va_pred = model.predict(X_val)
        
        # Calculate real-world RMSE dynamically adjusting for target scale
        tr_rmse = np.sqrt(mean_squared_error(y_tr_fit, tr_pred)) * (y_std if is_scaled else 1.0)
        va_rmse = np.sqrt(mean_squared_error(y_va_fit, va_pred)) * (y_std if is_scaled else 1.0)
        
        train_hist.append(tr_rmse)
        val_hist.append(va_rmse)
        
    # Return appropriately wrapped model if targets were scaled
    if is_scaled:
        return ScaledModelWrapper(model, y_mean, y_std), train_hist, val_hist
        
    return model, train_hist, val_hist

def predict_model(model, X_val):
    preds = model.predict(X_val)
    return np.maximum(preds, 0.0)

# ==========================================
# VISUALIZATION GENERATORS
# ==========================================
def plot_correlation_heatmap(X_df, y, images_dir):
    plot_df = X_df.copy()
    plot_df['Price($)'] = y
    
    corr = plot_df.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
    n_cols = len(corr.columns)
    fig_size = max(16, int(n_cols * 0.55))

    plt.figure(figsize=(fig_size, int(fig_size * 0.85)))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5, 
        cbar_kws={'shrink': 0.8},
        annot_kws={'size': max(6, 10 - n_cols // 5)}
    )
    plt.title('Complete Feature Correlation Heatmap (All Active Features vs. Price)', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

def plot_overfitting_charts(histories, X_tr_weighted, y_train, X_val_weighted, y_val, images_dir):
    for model_name, hist in histories.items():
        plt.figure(figsize=(8, 5))
        plt.plot(hist['train'], label='On training set', color='#2b5c8f', linewidth=2)
        plt.plot(hist['val'], label='On test set', color='#46a040', linewidth=2)
        plt.title(f'Overfitting Analysis: {model_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Epochs / Training Iterations', fontsize=10)
        plt.ylabel('RMSE Error ($)', fontsize=10)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name).lower()
        plt.savefig(os.path.join(images_dir, f'overfitting_{safe_name}.png'), dpi=300)
        plt.close()

    k_max = min(25, max(2, len(y_train) // 2))
    k_range = list(range(1, k_max))
    knn_tr_err, knn_val_err = [], []
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance', algorithm='kd_tree').fit(X_tr_weighted, y_train)
        knn_tr_err.append(np.sqrt(mean_squared_error(y_train, np.maximum(knn.predict(X_tr_weighted), 0.0))))
        knn_val_err.append(np.sqrt(mean_squared_error(y_val, np.maximum(knn.predict(X_val_weighted), 0.0))))

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, knn_tr_err, label='On training set', color='#2b5c8f', linewidth=2)
    plt.plot(k_range, knn_val_err, label='On test set', color='#46a040', linewidth=2)
    plt.title('Overfitting Analysis: k-NN Tree-Based (Sklearn)', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Neighbors (k)', fontsize=10)
    plt.ylabel('RMSE Error ($)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'overfitting_knn_kd_tree.png'), dpi=300)
    plt.close()

def plot_scatter_fits(X_val_scaled, y_val, predictions, feature_names, images_dir):
    sqft_idx = feature_names.index('areainfeet') if 'areainfeet' in feature_names else 0
    sqft_vals = X_val_scaled[:, sqft_idx]

    for model_name, y_pred in predictions.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(y_val, y_pred, alpha=0.4, color='#2b5c8f', edgecolors='none', s=25)
        max_val = max(np.max(y_val), np.max(y_pred)) if len(y_val) > 0 else 1.0
        axes[0].plot([0, max_val], [0, max_val], 'r--', label='Ideal Fit Line (y=x)', linewidth=2)
        axes[0].set_title(f'Actual vs Predicted Price ($)\n{model_name}', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Actual Listed Price ($)', fontsize=10)
        axes[0].set_ylabel('Predicted Fair Value ($)', fontsize=10)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].scatter(sqft_vals, y_val, alpha=0.3, color='gray', label='Actual Listings', s=20)
        sorted_indices = np.argsort(sqft_vals)
        axes[1].plot(sqft_vals[sorted_indices], y_pred[sorted_indices], color='purple', label='Model Regression Line', linewidth=2)
        axes[1].set_title(f'Square Footage vs Price Model Fit\n{model_name}', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Square Feet (Standardized)', fontsize=10)
        axes[1].set_ylabel('Price ($)', fontsize=10)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name).lower()
        plt.savefig(os.path.join(images_dir, f'scatter_fit_{safe_name}.png'), dpi=300)
        plt.close()

# ==========================================
# PIPELINE EXECUTION FOR DATASET / UNIT TYPE
# ==========================================
def process_unit_type(unittype_name, csv_filename, models_dir="models", images_dir="images"):
    print(f"\n==================================================")
    print(f"Processing Dataset: {unittype_name} ({csv_filename})")
    print(f"==================================================")
    
    if not os.path.exists(csv_filename):
        print(f" [Warning] File {csv_filename} not found. Skipping {unittype_name}.")
        return

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    (
        X_df, df_clean, y, binary_cols, 
        unit_dummy_cols, agreement_dummy_cols, median_area
    ) = load_and_preprocess_data(csv_filename)
    
    if len(y) < 10:
        print(f" [Warning] Insufficient data samples ({len(y)}) for {unittype_name}. Skipping.")
        return

    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)

    # Base Model Definitions for CV and Final Export
    # SGDRegressor represents the PyTorch Linear nodes identically over minibatches
    m_lin_base = SGDRegressor(loss='squared_error', penalty=None, learning_rate='invscaling', eta0=0.01, random_state=42)
    m_lin_scaled_base = SGDRegressor(loss='squared_error', penalty='l2', alpha=1e-4, learning_rate='invscaling', eta0=0.01, random_state=42)
    
    # MLPRegressor configurations directly mirror the PyTorch (128, 64) layered structure
    m_mlp_base = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', learning_rate_init=0.001, random_state=42)
    m_mlp_scaled_base = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.01, learning_rate_init=0.003, random_state=42)

    # STEP 1: CROSS-VALIDATION
    n_splits = min(5, len(y))
    if n_splits < 2:
        print(f" [Warning] Too few samples for CV in {unittype_name}. Skipping CV.")
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_models = {
            'Multiple Linear Regression': None,
            'k-NN Tree-Based: KD-Tree': None,
            'Feedforward Neural Network MLP (Sklearn)': None
        }
        
        oof_predictions = {name: np.zeros(len(y), dtype=np.float32) for name in cv_models.keys()}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            scaler_cv = StandardScaler()
            X_tr_sc = scaler_cv.fit_transform(X_tr)
            X_va_sc = scaler_cv.transform(X_va)

            X_tr_wt = apply_feature_weights(X_tr_sc, feature_names)
            X_va_wt = apply_feature_weights(X_va_sc, feature_names)

            # Multiple Linear Regression
            m_log, _, _ = train_sklearn_iterative(m_lin_base, X_tr_sc, y_tr, X_va_sc, y_va, epochs=150, batch_size=128, is_scaled=False)
            oof_predictions['Multiple Linear Regression'][val_idx] = predict_model(m_log, X_va_sc)

            # KNN
            knn_cv = KNeighborsRegressor(n_neighbors=min(7, max(1, len(X_tr)-1)), weights='distance', algorithm='kd_tree', n_jobs=-1).fit(X_tr_wt, y_tr)
            oof_predictions['k-NN Tree-Based: KD-Tree'][val_idx] = np.maximum(knn_cv.predict(X_va_wt), 0.0)

            # MLP Target-Scaled
            m_mlp, _, _ = train_sklearn_iterative(m_mlp_scaled_base, X_tr_sc, y_tr, X_va_sc, y_va, epochs=400, batch_size=128, is_scaled=True)
            oof_predictions['Feedforward Neural Network MLP (Sklearn)'][val_idx] = predict_model(m_mlp, X_va_sc)

        consensus_fair_value = (oof_predictions['k-NN Tree-Based: KD-Tree'] + oof_predictions['Feedforward Neural Network MLP (Sklearn)']) / 2.0
        y_true_class = classify_valuation(y, consensus_fair_value, threshold=0.10)

        print("\n" + "=" * 80)
        print("FINAL EVALUATION METRICS & CONFUSION MATRICES ACROSS ALL K-FOLDS")
        print("=" * 80)
        for model_name, oof_preds in oof_predictions.items():
            mae = np.mean(np.abs(y - oof_preds))
            y_pred_class = classify_valuation(y, oof_preds, threshold=0.10)
            
            cm = confusion_matrix(y_true_class, y_pred_class)
            report = classification_report(
                y_true_class, 
                y_pred_class, 
                target_names=['Undervalued (0)', 'Fairly Valued (1)', 'Overvalued (2)'],
                digits=3,
                zero_division=0
            )

            print(f"\nModel: {model_name}")
            print(f"Mean Absolute Valuation Error (Magnitude): ${mae:.2f}")
            print("-" * 55)
            print("Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
            print(f"{'':21}| {'Pred Under (0)':14} | {'Pred Fair (1)':14} | {'Pred Over (2)':14}")
            print("-" * 72)
            row_labels = ["Undervalued (0)", "Fairly Valued (1)", "Overvalued (2)"]
            for idx, row in enumerate(cm):
                print(f"{row_labels[idx]:21}| {row[0]:<14} | {row[1]:<14} | {row[2]:<14}")
            print("-" * 72)
            print(report)

    # STEP 2: TRAINING FINAL MODELS ON FULL DATASET & EXPORTS
    print(f"\n--- Training Final Models & Exporting Artifacts for {unittype_name} ---")
    plot_correlation_heatmap(X_df, y, images_dir)

    test_size = 0.2 if len(X) >= 10 else 0.5
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    X_tr_weighted = apply_feature_weights(X_tr_scaled, feature_names)
    X_val_weighted = apply_feature_weights(X_val_scaled, feature_names)

    histories = {}

    model_lin_orig, tr_h1, val_h1 = train_sklearn_iterative(m_lin_base, X_tr_scaled, y_tr, X_val_scaled, y_val, epochs=150, is_scaled=False)
    pred_lin_orig = predict_model(model_lin_orig, X_val_scaled)
    histories['Multiple Linear Regression'] = {'train': tr_h1, 'val': val_h1}

    model_lin_scaled, tr_h2, val_h2 = train_sklearn_iterative(m_lin_scaled_base, X_tr_scaled, y_tr, X_val_scaled, y_val, epochs=200, is_scaled=True)
    pred_lin_scaled = predict_model(model_lin_scaled, X_val_scaled)
    histories['Multiple Linear Regression (Target-Scaled)'] = {'train': tr_h2, 'val': val_h2}

    n_neighbors = min(7, max(1, len(X_tr)-1))
    knn_kd = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='kd_tree', n_jobs=-1).fit(X_tr_weighted, y_tr)
    pred_kd = np.maximum(knn_kd.predict(X_val_weighted), 0.0)

    model_mlp_orig, tr_h4, val_h4 = train_sklearn_iterative(m_mlp_base, X_tr_scaled, y_tr, X_val_scaled, y_val, epochs=500, is_scaled=False)
    pred_mlp_orig = predict_model(model_mlp_orig, X_val_scaled)
    histories['Feedforward Neural Network MLP'] = {'train': tr_h4, 'val': val_h4}

    model_mlp_scaled, tr_h5, val_h5 = train_sklearn_iterative(m_mlp_scaled_base, X_tr_scaled, y_tr, X_val_scaled, y_val, epochs=400, is_scaled=True)
    pred_mlp_scaled = predict_model(model_mlp_scaled, X_val_scaled)
    histories['Feedforward Neural Network MLP (Target-Scaled)'] = {'train': tr_h5, 'val': val_h5}

    predictions = {
        'Multiple Linear Regression': pred_lin_orig,
        'Multiple Linear Regression (Target-Scaled)': pred_lin_scaled,
        'k-NN Tree-Based: KD-Tree': pred_kd,
        'Feedforward Neural Network MLP': pred_mlp_orig,
        'Feedforward Neural Network MLP (Target-Scaled)': pred_mlp_scaled
    }

    plot_overfitting_charts(histories, X_tr_weighted, y_tr, X_val_weighted, y_val, images_dir)
    plot_scatter_fits(X_val_scaled, y_val, predictions, feature_names, images_dir)

    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    X_weighted_full = apply_feature_weights(X_scaled_full, feature_names)
    joblib.dump(scaler_full, os.path.join(models_dir, 'scaler.pkl'))

    meta = {
        'unittype': unittype_name,
        'feature_names': feature_names,
        'amenity_cols': binary_cols,
        'unit_dummy_cols': unit_dummy_cols,
        'agreement_dummy_cols': agreement_dummy_cols,
        'target_location': [43.651070, -79.347015],
        'median_area': float(median_area),
        'models_map': {
            'Multiple Linear Regression': 'multiple_linear_original_sklearn.joblib',
            'Multiple Linear Regression (Target-Scaled)': 'multiple_linear_scaled_sklearn.joblib',
            'k-NN Tree-Based: KD-Tree': 'knn_kd_tree.joblib',
            'Feedforward Neural Network MLP': 'mlp_original_sklearn.joblib',
            'Feedforward Neural Network MLP (Target-Scaled)': 'mlp_scaled_sklearn.joblib'
        }
    }
    with open(os.path.join(models_dir, 'feature_meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # Persist via joblib rather than Torch JIT
    joblib.dump(model_lin_orig, os.path.join(models_dir, 'multiple_linear_original_sklearn.joblib'))
    joblib.dump(model_lin_scaled, os.path.join(models_dir, 'multiple_linear_scaled_sklearn.joblib'))
    joblib.dump(KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm='kd_tree').fit(X_weighted_full, y), os.path.join(models_dir, 'knn_kd_tree.joblib'))
    joblib.dump(model_mlp_orig, os.path.join(models_dir, 'mlp_original_sklearn.joblib'))
    joblib.dump(model_mlp_scaled, os.path.join(models_dir, 'mlp_scaled_sklearn.joblib'))

    print(f" Completed {unittype_name}: Models saved in '{models_dir}/', Visuals saved in '{images_dir}/'")

# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    if os.path.exists('total_cleaned.csv'):
        process_unit_type('Total', 'total_cleaned.csv', models_dir='models', images_dir='images')

    unit_type_files = {
        'Apartment': 'total_cleaned_apartment.csv',
        'Basement': 'total_cleaned_basement.csv',
        'Condo': 'total_cleaned_condo.csv',
        'Duplex_Triplex': 'total_cleaned_duplex_triplex.csv',
        'House': 'total_cleaned_house.csv',
        'Townhouse': 'total_cleaned_townhouse.csv'
    }

    for unittype_name, csv_filename in unit_type_files.items():
        if os.path.exists(csv_filename):
            models_dir = f"model_{unittype_name.lower()}"
            images_dir = f"image_{unittype_name.lower()}"
            process_unit_type(unittype_name, csv_filename, models_dir=models_dir, images_dir=images_dir)

    print("\nAll pipeline tasks executed successfully!")

if __name__ == '__main__':
    main()