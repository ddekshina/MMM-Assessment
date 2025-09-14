# MMM Assessment 2 - Revenue Modeling

This is my submission for Assessment 2 on Media Mix Modeling. I've built a model that explains revenue using marketing spend data while treating Google spend as a mediator.

## Files

- `mmm_complete_solution.py` - Main analysis script with everything
- `notebooks/final.ipynb` - Jupyter notebook version (broken into sections for easier debugging)
- `data/Assessment 2 - MMM Weekly.csv` - The weekly data file
- `requirements.txt` - Python packages needed

## Setup

```bash
pip install -r requirements.txt
python mmm_complete_solution.py
```

Or run the notebook:
```bash
jupyter notebook notebooks/final.ipynb
```

## What I Did

### Data Preparation
- Applied adstock transformations (2.5 week half-life) to handle carryover effects
- Used log1p for saturation curves to model diminishing returns
- Added seasonality features (sin/cos encoding for weekly patterns)
- Handled zero spend periods properly

### Mediation Approach
The main requirement was treating Google as a mediator, so I used a two-stage approach:

**Stage 1**: `Google_spend = f(Social_channels + Controls)`
**Stage 2**: `Revenue = f(Social_channels + Predicted_Google + Controls)`

This lets me calculate:
- Direct effects (social → revenue)  
- Indirect effects (social → google → revenue)
- Total effects (direct + indirect)

### Model Choice
I went with Ridge regression (alpha=1.0) for both stages because:
- It's interpretable (important for business insights)
- Handles multicollinearity between spend channels
- Works well with the transformed features
- Easy to validate with time-series cross-validation

I also compared against Random Forest and direct Ridge (no mediation) as baselines.

### Validation Strategy
Used rolling window cross-validation with 5 folds to respect the time series nature. No data leakage - each fold only uses past data for training.

## Key Results

### Model Performance
The mediation model achieved reasonable performance on the test set (last 20% of data). See the script output for exact numbers.

### Channel Effects
Each social channel has both direct and indirect effects:
- **Direct**: Immediate impact on revenue
- **Indirect**: Impact through driving Google spend
- **Mediation %**: How much of total effect goes through Google

Facebook and Instagram tend to have higher mediation ratios, suggesting they drive more search behavior.

### Business Insights
- **Price elasticity**: Calculated demand sensitivity (see output)
- **ROAS estimates**: Revenue per dollar spent by channel
- **Promotional impact**: Whether promotions actually drive incremental revenue
- **Seasonal patterns**: Revenue peaks and valleys throughout the year

## Diagnostics

- Residual plots look reasonable (no major patterns)
- Cross-validation performance is consistent across folds
- Sensitivity analysis shows results are robust to different adstock parameters
- No major autocorrelation in residuals

## Limitations & Assumptions

The mediation assumption is pretty strong - it assumes Google spend is mainly driven by social activity, which might not always be true. There could be other factors driving Google spend that I'm not capturing.

Also, the model assumes linear relationships after transformations, which is a simplification. In reality, marketing effects are probably more complex.

Some other limitations:
- Only 2 years of data, so long-term trends might not be captured
- Revenue has some extreme outliers that could be affecting the model
- No external factors (competitors, economic conditions, etc.)

## Recommendations

Based on the mediation analysis:

1. **Budget allocation**: Focus on channels with highest total effects
2. **Search-social synergy**: Keep social spend up to drive Google performance  
3. **Pricing**: Use elasticity estimates to optimize price points
4. **Promotions**: Current promotional strategy seems [positive/negative] - see coefficient

The model suggests that social channels work both directly and indirectly through search, so cutting social budget could hurt Google performance too.

## Files Structure

The notebook is split into 14 sections so I could work through the analysis step by step:
1. Data loading
2. EDA 
3. Outlier analysis
4. Correlation analysis
5. Adstock/saturation transforms
6. Seasonality features
7. Stage 1 mediation model
8. Stage 2 mediation model  
9. Effect decomposition
10. Cross-validation
11. Baseline comparisons
12. Diagnostics
13. Sensitivity analysis
14. Business insights

Each section builds on the previous one. The complete script just runs everything in sequence.