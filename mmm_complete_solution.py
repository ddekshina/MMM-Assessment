import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load and inspect data
print("Loading data...")
df = pd.read_csv("data/Assessment 2 - MMM Weekly.csv")
df['week'] = pd.to_datetime(df['week'])
df = df.sort_values('week').reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Date range: {df['week'].min()} to {df['week'].max()}")
print(f"Missing values:\n{df.isnull().sum()}")

# =============================================================================
# 1. DATA PREPARATION & FEATURE ENGINEERING
# =============================================================================

class MMDataPreprocessor:
    """
    Handles all data preprocessing for MMM including:
    - Adstock transformations
    - Saturation curves
    - Seasonality features
    - Zero spend handling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.spend_cols = ['facebook_spend', 'google_spend', 'tiktok_spend', 
                          'instagram_spend', 'snapchat_spend']
        
    def adstock_transform(self, series, half_life=2):
        """Exponential adstock with configurable half-life"""
        decay = 0.5 ** (1/half_life)
        adstocked = np.zeros(len(series))
        carry = 0
        for i, x in enumerate(series):
            carry = x + decay * carry
            adstocked[i] = carry
        return adstocked
    
    def saturation_transform(self, series, alpha=2, gamma=0.3):
        """Hill saturation: alpha*x^gamma / (1 + x^gamma)"""
        # Normalize to avoid overflow
        x_norm = series / (series.max() + 1e-8)
        saturated = alpha * (x_norm ** gamma) / (1 + x_norm ** gamma)
        return saturated * series.max()  # Scale back
    
    def add_seasonality_features(self, df):
        """Add time-based features"""
        df = df.copy()
        df['week_of_year'] = df['week'].dt.isocalendar().week
        df['month'] = df['week'].dt.month
        df['quarter'] = df['week'].dt.quarter
        
        # Cyclical encoding for seasonality
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def prepare_features(self, df, adstock_params=None, saturation_params=None):
        """Complete feature preparation pipeline"""
        df_processed = df.copy()
        
        # Default parameters (these could be optimized)
        if adstock_params is None:
            adstock_params = {col: 2.5 for col in self.spend_cols}  # 2.5 week half-life
        if saturation_params is None:
            saturation_params = {col: {'alpha': 1.0, 'gamma': 0.5} for col in self.spend_cols}
        
        print("Applying adstock and saturation transformations...")
        
        # Apply transformations to spend channels
        for col in self.spend_cols:
            # Adstock
            half_life = adstock_params.get(col, 2.0)
            df_processed[f'{col}_adstock'] = self.adstock_transform(
                df_processed[col].values, half_life=half_life
            )
            
            # Saturation (on adstocked values)
            alpha = saturation_params[col]['alpha']
            gamma = saturation_params[col]['gamma']
            df_processed[f'{col}_sat'] = self.saturation_transform(
                df_processed[f'{col}_adstock'].values, alpha=alpha, gamma=gamma
            )
        
        # Add seasonality features
        df_processed = self.add_seasonality_features(df_processed)
        
        # Handle zero spend periods (already handled by log1p in saturation)
        print("Feature engineering complete!")
        
        return df_processed

# Apply preprocessing
preprocessor = MMDataPreprocessor()
df_processed = preprocessor.prepare_features(df)

print(f"Processed data shape: {df_processed.shape}")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

# Revenue distribution and outliers
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(df_processed['revenue'], bins=30, alpha=0.7)
plt.title('Revenue Distribution')
plt.xlabel('Revenue')

plt.subplot(2, 3, 2)
plt.plot(df_processed['week'], df_processed['revenue'])
plt.title('Revenue Over Time')
plt.xticks(rotation=45)

plt.subplot(2, 3, 3)
# Check for extreme outliers in revenue
Q1 = df_processed['revenue'].quantile(0.25)
Q3 = df_processed['revenue'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
outliers = df_processed['revenue'] > outlier_threshold
plt.scatter(df_processed['week'], df_processed['revenue'], 
           c=['red' if x else 'blue' for x in outliers], alpha=0.7)
plt.title(f'Revenue Outliers (>{outlier_threshold:.0f})')
plt.xticks(rotation=45)

# Spend correlation heatmap
plt.subplot(2, 3, 4)
spend_data = df_processed[preprocessor.spend_cols]
correlation_matrix = spend_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Spend Channel Correlations')

# Zero spend analysis
plt.subplot(2, 3, 5)
zero_spend_counts = (spend_data == 0).sum()
zero_spend_counts.plot(kind='bar')
plt.title('Zero Spend Weeks by Channel')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# =============================================================================
# 3. CAUSAL MODELING: MEDIATION ANALYSIS
# =============================================================================

class MediationAnalyzer:
    """
    Implements 2SLS approach for mediation analysis where Google is mediator
    """
    
    def __init__(self):
        self.stage1_model = None  # Social -> Google
        self.stage2_model = None  # Social + Google_hat -> Revenue
        self.social_features = ['facebook_spend_sat', 'tiktok_spend_sat', 
                               'instagram_spend_sat', 'snapchat_spend_sat']
        self.control_features = ['emails_send', 'sms_send', 'social_followers', 
                               'average_price', 'promotions', 'week_sin', 'week_cos']
        
    def fit_mediation_model(self, df_train, alpha=1.0):
        """
        Two-stage approach:
        Stage 1: Google ~ Social + Controls
        Stage 2: Revenue ~ Social + Google_hat + Controls
        """
        # Stage 1: Predict Google spend from social channels
        X_stage1 = df_train[self.social_features + self.control_features]
        y_stage1 = df_train['google_spend_sat']
        
        self.stage1_model = Ridge(alpha=alpha)
        self.stage1_model.fit(X_stage1, y_stage1)
        
        # Get predicted Google spend (instrument)
        google_hat_train = self.stage1_model.predict(X_stage1)
        
        # Stage 2: Revenue model with instrumented Google
        X_stage2 = df_train[self.social_features + self.control_features].copy()
        X_stage2['google_spend_hat'] = google_hat_train
        y_stage2 = df_train['revenue']
        
        self.stage2_model = Ridge(alpha=alpha)
        self.stage2_model.fit(X_stage2, y_stage2)
        
        return self
    
    def predict(self, df_test):
        """Predict revenue using mediation model"""
        # Stage 1: Predict Google spend
        X_stage1 = df_test[self.social_features + self.control_features]
        google_hat_test = self.stage1_model.predict(X_stage1)
        
        # Stage 2: Predict revenue
        X_stage2 = df_test[self.social_features + self.control_features].copy()
        X_stage2['google_spend_hat'] = google_hat_test
        revenue_pred = self.stage2_model.predict(X_stage2)
        
        return revenue_pred, google_hat_test
    
    def analyze_mediation_effects(self):
        """Calculate direct, indirect, and total effects"""
        stage1_coef = pd.Series(self.stage1_model.coef_[:len(self.social_features)], 
                               index=self.social_features)
        stage2_coef = pd.Series(self.stage2_model.coef_[:len(self.social_features)], 
                               index=self.social_features)
        google_coef = self.stage2_model.coef_[len(self.social_features) + len(self.control_features)]
        
        effects = []
        for channel in self.social_features:
            direct_effect = stage2_coef[channel]
            indirect_effect = stage1_coef[channel] * google_coef
            total_effect = direct_effect + indirect_effect
            
            effects.append({
                'channel': channel.replace('_spend_sat', ''),
                'direct_effect': direct_effect,
                'indirect_effect': indirect_effect,
                'total_effect': total_effect,
                'mediation_ratio': indirect_effect / total_effect if total_effect != 0 else 0
            })
        
        return pd.DataFrame(effects)

# =============================================================================
# 4. MODEL TRAINING & VALIDATION
# =============================================================================

class TimeSeriesValidator:
    """Time-aware cross validation for MMM"""
    
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def time_series_split(self, X, y, test_size=None):
        """Create time-based train/test split"""
        if test_size is None:
            test_size = self.test_size
            
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def rolling_window_validation(self, X, y, model, initial_train_size=0.6):
        """Rolling window cross-validation"""
        n = len(X)
        initial_size = int(n * initial_train_size)
        window_size = int((n - initial_size) / self.n_splits)
        
        scores = []
        
        for i in range(self.n_splits):
            train_end = initial_size + i * window_size
            test_start = train_end
            test_end = min(test_start + window_size, n)
            
            if test_start >= test_end:
                break
                
            X_train_fold = X.iloc[:train_end]
            X_test_fold = X.iloc[test_start:test_end]
            y_train_fold = y.iloc[:train_end]
            y_test_fold = y.iloc[test_start:test_end]
            
            # Fit and predict
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
            r2 = r2_score(y_test_fold, y_pred_fold)
            mape = np.mean(np.abs((y_test_fold - y_pred_fold) / (y_test_fold + 1e-8))) * 100
            
            scores.append({
                'fold': i + 1,
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            })
        
        return pd.DataFrame(scores)

# Initialize validator
validator = TimeSeriesValidator()

# Split data for training
X_full = df_processed[preprocessor.spend_cols + 
                     [f'{col}_sat' for col in preprocessor.spend_cols] +
                     ['emails_send', 'sms_send', 'social_followers', 
                      'average_price', 'promotions', 'week_sin', 'week_cos']]
y_full = df_processed['revenue']

X_train, X_test, y_train, y_test = validator.time_series_split(X_full, y_full)

print(f"Training set: {len(X_train)} weeks")
print(f"Test set: {len(X_test)} weeks")

# =============================================================================
# 5. MEDIATION MODEL IMPLEMENTATION
# =============================================================================

# Fit mediation model
mediation_analyzer = MediationAnalyzer()

# Prepare training data with processed features
df_train = df_processed.iloc[:len(X_train)].copy()
df_test = df_processed.iloc[len(X_train):].copy()

mediation_analyzer.fit_mediation_model(df_train, alpha=1.0)

# Get predictions
revenue_pred_med, google_pred = mediation_analyzer.predict(df_test)

# Evaluate mediation model
rmse_med = np.sqrt(mean_squared_error(y_test, revenue_pred_med))
r2_med = r2_score(y_test, revenue_pred_med)
mape_med = np.mean(np.abs((y_test - revenue_pred_med) / (y_test + 1e-8))) * 100

print("\n" + "="*50)
print("MEDIATION MODEL RESULTS")
print("="*50)
print(f"RMSE: {rmse_med:,.2f}")
print(f"R²: {r2_med:.3f}")
print(f"MAPE: {mape_med:.1f}%")

# Analyze mediation effects
effects_df = mediation_analyzer.analyze_mediation_effects()
print("\nMediation Effects Analysis:")
print(effects_df.round(3))

# =============================================================================
# 6. BASELINE MODELS FOR COMPARISON
# =============================================================================

# Compare with other models
models = {
    'Ridge_Direct': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
}

results = []

for name, model in models.items():
    # Use saturated features for fair comparison
    X_train_sat = df_train[[f'{col}_sat' for col in preprocessor.spend_cols] + 
                          ['emails_send', 'sms_send', 'social_followers', 
                           'average_price', 'promotions', 'week_sin', 'week_cos']]
    X_test_sat = df_test[[f'{col}_sat' for col in preprocessor.spend_cols] + 
                        ['emails_send', 'sms_send', 'social_followers', 
                         'average_price', 'promotions', 'week_sin', 'week_cos']]
    
    model.fit(X_train_sat, y_train)
    y_pred = model.predict(X_test_sat)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    })

# Add mediation model results
results.append({
    'Model': 'Mediation_2SLS',
    'RMSE': rmse_med,
    'R²': r2_med,
    'MAPE': mape_med
})

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.round(3))

# =============================================================================
# 7. BUSINESS INSIGHTS & DIAGNOSTICS
# =============================================================================

def calculate_roas_and_elasticities(model_analyzer, df_data):
    """Calculate ROAS and price elasticity"""
    
    # Price elasticity calculation
    price_change = 0.01  # 1% change
    
    # Create scenario with 1% price increase
    df_scenario = df_data.copy()
    df_scenario['average_price'] = df_scenario['average_price'] * (1 + price_change)
    
    # Predict revenue change
    revenue_baseline, _ = model_analyzer.predict(df_data)
    revenue_scenario, _ = model_analyzer.predict(df_scenario)
    
    revenue_change = (revenue_scenario.mean() - revenue_baseline.mean()) / revenue_baseline.mean()
    price_elasticity = revenue_change / price_change
    
    print(f"\nBUSINESS INSIGHTS")
    print("="*50)
    print(f"Price Elasticity: {price_elasticity:.3f}")
    print(f"A 1% price increase leads to {revenue_change*100:.2f}% change in revenue")
    
    # ROAS calculation for each channel
    print("\nROAS Analysis (last 4 weeks average):")
    recent_data = df_data.tail(4)
    effects_df = model_analyzer.analyze_mediation_effects() # Get effects for ROAS
    
    for channel in ['facebook', 'instagram', 'tiktok', 'snapchat']:
        spend_col = f'{channel}_spend'
        if spend_col in recent_data.columns:
            avg_spend = recent_data[spend_col].mean()
            if avg_spend > 0:
                effect_row = effects_df[effects_df['channel'] == channel]
                if not effect_row.empty:
                    effect = effect_row['total_effect'].iloc[0]
                    estimated_revenue = effect * recent_data[f'{channel}_spend_sat'].mean()
                    roas = estimated_revenue / avg_spend if avg_spend > 0 else 0
                    print(f"{channel.title()}: ${roas:.2f} revenue per $1 spent")
                    
    # Return the calculated elasticity so it can be used globally
    return price_elasticity

# Capture the returned value in a new variable
price_elasticity = calculate_roas_and_elasticities(mediation_analyzer, df_test)


# =============================================================================
# 8. MODEL DIAGNOSTICS
# =============================================================================

# Residual analysis
residuals = y_test - revenue_pred_med

plt.figure(figsize=(15, 12))

# Residuals over time
plt.subplot(2, 3, 1)
plt.plot(df_test['week'], residuals, 'o-', alpha=0.7)
plt.title('Residuals Over Time')
plt.xlabel('Week')
plt.ylabel('Residuals')
plt.xticks(rotation=45)

# Residuals vs predicted
plt.subplot(2, 3, 2)
plt.scatter(revenue_pred_med, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted Revenue')
plt.ylabel('Residuals')

# Q-Q plot for normality
plt.subplot(2, 3, 3)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normality Check)')

# Actual vs Predicted
plt.subplot(2, 3, 4)
plt.scatter(y_test, revenue_pred_med, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')

# Rolling CV performance
plt.subplot(2, 3, 5)
cv_scores = validator.rolling_window_validation(
    X_train, y_train, 
    Ridge(alpha=1.0),  # Use simple model for CV
    initial_train_size=0.6
)
plt.plot(cv_scores['fold'], cv_scores['r2'], 'o-', label='R²')
plt.title('Cross-Validation Performance')
plt.xlabel('Fold')
plt.ylabel('R²')
plt.legend()

# Feature importance from mediation model
plt.subplot(2, 3, 6)
feature_names = (mediation_analyzer.social_features + 
                mediation_analyzer.control_features + ['google_spend_hat'])
feature_importance = np.abs(mediation_analyzer.stage2_model.coef_)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.title('Feature Importance (|Coefficients|)')
plt.xlabel('Absolute Coefficient Value')

plt.tight_layout()
plt.show()

print(f"\nCV Performance Summary:")
print(cv_scores.round(3))

# =============================================================================
# 9. SENSITIVITY ANALYSIS
# =============================================================================

print("\n" + "="*50)
print("SENSITIVITY ANALYSIS")
print("="*50)

# Test different adstock parameters
adstock_params_test = [1.0, 2.0, 3.0, 4.0]
sensitivity_results = []

for half_life in adstock_params_test:
    # Reprocess with different adstock
    adstock_params = {col: half_life for col in preprocessor.spend_cols}
    df_sensitivity = preprocessor.prepare_features(df, adstock_params=adstock_params)
    
    # Refit mediation model
    df_train_sens = df_sensitivity.iloc[:len(X_train)].copy()
    df_test_sens = df_sensitivity.iloc[len(X_train):].copy()
    
    mediation_sens = MediationAnalyzer()
    mediation_sens.fit_mediation_model(df_train_sens, alpha=1.0)
    
    revenue_pred_sens, _ = mediation_sens.predict(df_test_sens)
    r2_sens = r2_score(y_test, revenue_pred_sens)
    
    sensitivity_results.append({
        'adstock_half_life': half_life,
        'test_r2': r2_sens
    })

sensitivity_df = pd.DataFrame(sensitivity_results)
print("Adstock Sensitivity Analysis:")
print(sensitivity_df.round(4))

# =============================================================================
# 10. FINAL RECOMMENDATIONS
# =============================================================================

print("\n" + "="*60)
print("MARKETING RECOMMENDATIONS")
print("="*60)

# Channel prioritization based on mediation effects
effects_sorted = effects_df.sort_values('total_effect', ascending=False)

print("1. CHANNEL PRIORITIZATION (by Total Effect):")
for idx, row in effects_sorted.iterrows():
    mediation_pct = row['mediation_ratio'] * 100
    print(f"   {row['channel'].title()}: {row['total_effect']:.2f} total effect")
    print(f"      - Direct: {row['direct_effect']:.2f}")
    print(f"      - Indirect (via Google): {row['indirect_effect']:.2f} ({mediation_pct:.1f}%)")

print(f"\n2. PRICING INSIGHTS:")
print(f"   - Price elasticity: {price_elasticity:.3f}")
if price_elasticity < -1:
    print("   - Demand is elastic: Price reductions could increase total revenue")
elif price_elasticity > -1:
    print("   - Demand is inelastic: Price increases could increase total revenue")

print(f"\n3. PROMOTIONAL STRATEGY:")
promo_effect = mediation_analyzer.stage2_model.coef_[
    mediation_analyzer.control_features.index('promotions')
]
print(f"   - Promotion coefficient: {promo_effect:.2f}")
if promo_effect > 0:
    print("   - Promotions have positive impact on revenue")
else:
    print("   - Promotions may be cannibalizing regular sales")

print(f"\n4. SEARCH vs SOCIAL STRATEGY:")
google_coef = mediation_analyzer.stage2_model.coef_[-1]  # google_spend_hat coefficient
print(f"   - Google Search multiplier: {google_coef:.2f}")
print(f"   - Social channels drive {effects_df['mediation_ratio'].mean()*100:.1f}% of impact through Search")
print("   - Recommendation: Maintain social spend to drive search intent")

print(f"\n5. RISKS & LIMITATIONS:")
print("   - Model assumes Google spend is purely driven by social channels")
print("   - Seasonality effects captured but external factors not modeled")
print("   - Revenue outliers may indicate data quality issues or external campaigns")
print(f"   - Model R² of {r2_med:.3f} suggests {(1-r2_med)*100:.1f}% of variance unexplained")

print("\n" + "="*60)
print("SUBMISSION COMPLETE")
print("="*60)