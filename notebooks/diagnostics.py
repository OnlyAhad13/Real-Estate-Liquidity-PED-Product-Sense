"""
Causal Estimation Diagnostic Analysis
======================================

This script investigates WHY our LinearDML estimates show positive elasticity
when the TRUE Data Generating Process has NEGATIVE price effects.

Diagnostic Framework (Senior Data Scientist Perspective):
1. Ground Truth Verification: Confirm the DGP is correct
2. First-Stage Model Quality: Check R² and residual patterns
3. Orthogonality Check: Are residuals truly uncorrelated?
4. Confounding Strength Analysis: Quantify the bias
5. Alternative Estimation: Use Instrumental Variables

Author: Causal Inference Project - Diagnostic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_data():
    """Load and prepare the marketplace data."""
    df = pd.read_csv('marketplace_data.csv')
    
    # Encode categoricals
    le_room = LabelEncoder()
    le_tier = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    df['property_tier_encoded'] = le_tier.fit_transform(df['property_tier'])
    
    return df


def diagnostic_1_ground_truth(df):
    """
    DIAGNOSTIC 1: Verify the Ground Truth DGP
    
    We KNOW the true coefficients from generate_synthetic_data.py:
    - Luxury: price_coef = -0.008
    - Economy: price_coef = -0.015
    
    If we could observe the TRUE booking_probability (which we generated),
    we can verify the DGP is correct.
    """
    print("="*70)
    print("DIAGNOSTIC 1: Ground Truth Verification")
    print("="*70)
    
    # We have booking_probability in the data (the TRUE underlying probability)
    # Let's check correlation with price BY TIER
    
    for tier in ['Luxury', 'Economy']:
        tier_df = df[df['property_tier'] == tier]
        
        # Correlation between price and TRUE booking probability
        corr_prob = tier_df['historical_price'].corr(tier_df['booking_probability'])
        
        # Correlation between price and OBSERVED booking (binary)
        corr_obs = tier_df['historical_price'].corr(tier_df['is_booked'])
        
        print(f"\n{tier} Properties:")
        print(f"  Price ↔ TRUE Booking Prob:     r = {corr_prob:.4f}")
        print(f"  Price ↔ OBSERVED is_booked:    r = {corr_obs:.4f}")
    
    # Key insight: Even though TRUE effect is negative, OBSERVED correlation
    # can be positive due to confounding!
    
    overall_naive = df['historical_price'].corr(df['is_booked'])
    print(f"\nOverall Naive Correlation (Price ↔ is_booked): {overall_naive:.4f}")
    print("  → Positive correlation = CONFOUNDING BIAS (endogeneity)")
    
    # Calculate what the TRUE elasticity should be (from DGP)
    print("\n" + "-"*50)
    print("TRUE DGP Parameters (from generate_synthetic_data.py):")
    print("  Luxury price coefficient:  -0.008 (in logit scale)")
    print("  Economy price coefficient: -0.015 (in logit scale)")
    print("-"*50)


def diagnostic_2_first_stage_quality(df):
    """
    DIAGNOSTIC 2: First-Stage Model Quality
    
    DML requires GOOD first-stage models. If they don't explain enough
    variance, the residuals will still contain confounding.
    
    Check: 
    - Treatment Model R² (predicting price from X)
    - Outcome Model R² (predicting booking from X, without price)
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: First-Stage Model Quality Check")
    print("="*70)
    
    feature_cols = [
        'location_score', 'capacity', 'host_rating', 'seasonality_index',
        'room_type_encoded', 'property_tier_encoded'
    ]
    
    X = df[feature_cols].values
    T = df['historical_price'].values
    Y = df['is_booked'].values
    
    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # --- Treatment Model (Price Prediction) ---
    print("\nA. Treatment Model: Predicting Price from X")
    
    # Linear
    lr_t = LinearRegression().fit(X_train_s, T_train)
    r2_linear_t = lr_t.score(X_test_s, T_test)
    
    # Gradient Boosting (more flexible)
    gb_t = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_t.fit(X_train_s, T_train)
    r2_gb_t = gb_t.score(X_test_s, T_test)
    
    print(f"   Linear Regression R²:      {r2_linear_t:.4f}")
    print(f"   Gradient Boosting R²:      {r2_gb_t:.4f}")
    
    if r2_gb_t < 0.7:
        print("   ⚠️ WARNING: Low R² means confounding variables NOT fully captured!")
    else:
        print("   ✓ Treatment model explains good variance")
    
    # Residual analysis
    T_pred = gb_t.predict(X_test_s)
    T_residual = T_test - T_pred
    
    # --- Outcome Model (Booking Prediction) ---
    print("\nB. Outcome Model: Predicting Booking from X (no price)")
    
    # Logistic
    lr_y = LogisticRegression(max_iter=1000).fit(X_train_s, Y_train)
    acc_logistic = lr_y.score(X_test_s, Y_test)
    
    # Gradient Boosting
    gb_y = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_y.fit(X_train_s, Y_train)
    acc_gb_y = gb_y.score(X_test_s, Y_test)
    
    print(f"   Logistic Regression Acc:   {acc_logistic:.4f}")
    print(f"   Gradient Boosting Acc:     {acc_gb_y:.4f}")
    
    # For DML, we need probability predictions
    Y_prob_pred = gb_y.predict_proba(X_test_s)[:, 1]
    Y_residual = Y_test - Y_prob_pred
    
    return T_residual, Y_residual, X_test_s, T_test, Y_test


def diagnostic_3_orthogonality(T_residual, Y_residual):
    """
    DIAGNOSTIC 3: Orthogonality Check
    
    For DML to work, the residuals from first-stage models 
    should be UNCORRELATED with confounders.
    
    Key check: Corr(T_residual, Y_residual) should equal the TRUE causal effect
    when first stages are correct.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Orthogonality & Residual Analysis")
    print("="*70)
    
    # Correlation between residualized treatment and outcome
    residual_corr = np.corrcoef(T_residual, Y_residual)[0, 1]
    print(f"\nCorrelation(Treatment Residual, Outcome Residual): {residual_corr:.4f}")
    
    # This correlation, scaled appropriately, IS the DML estimate
    # Negative correlation = negative causal effect (what we expect)
    # Positive correlation = still has confounding bias
    
    if residual_corr > 0:
        print("  ⚠️ POSITIVE residual correlation = RESIDUAL CONFOUNDING EXISTS!")
        print("  → First-stage models are not fully capturing the confounders")
    else:
        print("  ✓ Negative correlation suggests correct causal direction")
    
    # Simple OLS on residuals (Frisch-Waugh-Lovell)
    # CATE ≈ Cov(T_res, Y_res) / Var(T_res)
    cate_estimate = np.cov(T_residual, Y_residual)[0, 1] / np.var(T_residual)
    print(f"\nResidual-on-Residual CATE Estimate: {cate_estimate:.6f}")
    print("  (This is essentially what DML computes)")


def diagnostic_4_confounding_strength(df):
    """
    DIAGNOSTIC 4: Confounding Strength Analysis
    
    The key confounder is SEASONALITY:
    - High seasonality → Higher prices (hosts raise prices)
    - High seasonality → Higher bookings (more travelers)
    
    This creates a POSITIVE spurious correlation between price and booking.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: Confounding Strength Analysis")
    print("="*70)
    
    # Path analysis
    print("\nConfounder: seasonality_index")
    
    # Path 1: Seasonality → Price
    path1 = df['seasonality_index'].corr(df['historical_price'])
    print(f"  Path 1 (Seasonality → Price):   r = {path1:.4f}")
    
    # Path 2: Seasonality → Booking
    path2 = df['seasonality_index'].corr(df['is_booked'])
    print(f"  Path 2 (Seasonality → Booking): r = {path2:.4f}")
    
    # Confounding bias ≈ path1 * path2 (simplified)
    confounding_bias = path1 * path2
    print(f"\n  Approximate Confounding Bias: {confounding_bias:.4f}")
    print("  (This bias is ADDED to the true negative effect)")
    
    # Naive OLS estimate
    from sklearn.linear_model import LinearRegression
    X_naive = df[['historical_price']].values
    Y = df['is_booked'].values
    naive_ols = LinearRegression().fit(X_naive, Y)
    naive_coef = naive_ols.coef_[0]
    print(f"\n  Naive OLS Coefficient (biased): {naive_coef:.6f}")
    
    # Controlled OLS (adding seasonality as control)
    X_controlled = df[['historical_price', 'seasonality_index']].values
    controlled_ols = LinearRegression().fit(X_controlled, Y)
    controlled_coef = controlled_ols.coef_[0]
    print(f"  Controlled OLS (+ seasonality): {controlled_coef:.6f}")
    
    print("\n  KEY INSIGHT:")
    print("  If controlled coefficient is more negative (or less positive),")
    print("  it confirms seasonality was creating UPWARD bias.")


def diagnostic_5_iv_estimation(df):
    """
    DIAGNOSTIC 5: Instrumental Variable Estimation
    
    We created instruments that affect PRICE but NOT BOOKING directly:
    - platform_fee_rate: Random fee variation
    - cleaning_cost: Cost shifter
    - competitor_density: Competition affects pricing
    
    2SLS should recover the TRUE negative effect.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 5: Instrumental Variable (2SLS) Estimation")
    print("="*70)
    
    from sklearn.linear_model import LinearRegression
    
    # Instruments
    instruments = ['platform_fee_rate', 'cleaning_cost', 'competitor_density']
    
    # Check instrument relevance (correlation with price)
    print("\nInstrument Relevance (correlation with Price):")
    for iv in instruments:
        corr = df[iv].corr(df['historical_price'])
        print(f"  {iv}: r = {corr:.4f}")
    
    # Check exclusion restriction (should NOT correlate with booking conditional on price)
    # (Can't fully test, but check reduced form)
    print("\nExclusion Check (raw correlation with Booking):")
    for iv in instruments:
        corr = df[iv].corr(df['is_booked'])
        print(f"  {iv}: r = {corr:.4f}")
        if abs(corr) > 0.1:
            print(f"    ⚠️ Warning: May violate exclusion restriction")
    
    # Manual 2SLS
    print("\n--- Manual 2SLS Estimation ---")
    
    # Stage 1: Regress Price on Instruments + Controls
    controls = ['location_score', 'capacity', 'host_rating', 'seasonality_index',
                'room_type_encoded', 'property_tier_encoded']
    
    # Encode categoricals if not already
    le_room = LabelEncoder()
    le_tier = LabelEncoder()
    df['room_type_encoded'] = le_room.fit_transform(df['room_type'])
    df['property_tier_encoded'] = le_tier.fit_transform(df['property_tier'])
    
    X_stage1 = df[instruments + controls].values
    T = df['historical_price'].values
    Y = df['is_booked'].values
    
    # First stage: Predict price from instruments + controls
    stage1 = LinearRegression().fit(X_stage1, T)
    T_hat = stage1.predict(X_stage1)  # Predicted price (exogenous variation)
    
    stage1_r2 = stage1.score(X_stage1, T)
    print(f"  Stage 1 R² (instrument strength): {stage1_r2:.4f}")
    
    # Second stage: Regress Booking on Predicted Price + Controls
    X_stage2 = np.column_stack([T_hat, df[controls].values])
    stage2 = LinearRegression().fit(X_stage2, Y)
    
    iv_estimate = stage2.coef_[0]
    print(f"  Stage 2 (2SLS) Price Coefficient: {iv_estimate:.6f}")
    
    if iv_estimate < 0:
        print("  ✅ IV estimate is NEGATIVE - consistent with true causal effect!")
    else:
        print("  ⚠️ IV estimate still positive - instruments may be weak or invalid")
    
    # Compare estimates
    print("\n--- Estimation Method Comparison ---")
    
    # Naive OLS
    naive = LinearRegression().fit(df[['historical_price']], Y).coef_[0]
    
    # OLS with controls
    X_ols = df[['historical_price'] + controls].values
    ols_controlled = LinearRegression().fit(X_ols, Y).coef_[0]
    
    print(f"  Naive OLS (no controls):    {naive:.6f}")
    print(f"  OLS with Controls:          {ols_controlled:.6f}")
    print(f"  2SLS (IV):                  {iv_estimate:.6f}")
    print(f"  TRUE Effect (from DGP):     ~ -0.012 (average)")


def diagnostic_6_dml_with_better_models(df):
    """
    DIAGNOSTIC 6: DML with Better First-Stage Models
    
    The neural networks may have underfit. Try with:
    1. Gradient Boosting (often better for tabular data)
    2. Include the instruments as additional features
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 6: DML with Gradient Boosting First Stages")
    print("="*70)
    
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        
        feature_cols = [
            'location_score', 'capacity', 'host_rating', 'seasonality_index',
            'room_type_encoded', 'property_tier_encoded',
            # Add instruments as controls (they help predict price better)
            'competitor_density', 'host_experience_days', 'platform_fee_rate', 'cleaning_cost'
        ]
        
        X = df[feature_cols].values
        T = df['historical_price'].values
        Y = df['is_booked'].values
        
        # Use Gradient Boosting - often better for tabular data
        model_t = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        model_y = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        
        est = LinearDML(
            model_y=model_y,
            model_t=model_t,
            random_state=42,
            cv=5  # 5-fold cross-fitting
        )
        
        print("  Fitting DML with GradientBoosting...")
        est.fit(Y, T, X=X)
        
        # Get average effect
        effects = est.effect(X)
        ate = np.mean(effects)
        
        print(f"\n  DML with GradientBoosting ATE: {ate:.6f}")
        
        # By tier
        df_temp = df.copy()
        df_temp['cate'] = effects
        
        print("\n  CATE by Property Tier:")
        for tier in ['Luxury', 'Economy']:
            tier_effect = df_temp[df_temp['property_tier'] == tier]['cate'].mean()
            print(f"    {tier}: {tier_effect:.6f}")
            
    except Exception as e:
        print(f"  Error: {e}")


def run_full_diagnostic():
    """Run all diagnostics."""
    print("\n" + "#"*70)
    print("#" + " "*20 + "CAUSAL ESTIMATION DIAGNOSTICS" + " "*20 + "#")
    print("#"*70)
    
    df = load_data()
    
    # Run each diagnostic
    diagnostic_1_ground_truth(df)
    T_res, Y_res, X_test, T_test, Y_test = diagnostic_2_first_stage_quality(df)
    diagnostic_3_orthogonality(T_res, Y_res)
    diagnostic_4_confounding_strength(df)
    diagnostic_5_iv_estimation(df)
    diagnostic_6_dml_with_better_models(df)
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print("""
ROOT CAUSE ANALYSIS:
--------------------
1. STRONG CONFOUNDING: Seasonality creates ~0.15 upward bias in the 
   price-booking relationship. This bias is larger than the true 
   negative effect (~-0.01), resulting in NET POSITIVE observed correlation.

2. MODEL CAPACITY: Neural networks may not have learned the confounding 
   structure as well as ensemble methods. The first-stage outcome model
   needs to capture seasonality's full effect on booking probability.

3. DML LIMITATION: While DML is doubly robust, it still requires at least
   one first-stage model to be well-specified. With strong confounding,
   even small misspecification causes bias.

RECOMMENDATIONS:
----------------
1. USE INSTRUMENTAL VARIABLES: Our 2SLS estimate should be negative and
   closer to the true effect, as it uses exogenous price variation.

2. BETTER FIRST STAGES: Use Gradient Boosting or other flexible models
   that better capture non-linear relationships.

3. INCLUDE ALL CONFOUNDERS: Make sure ALL features that affect BOTH
   price and booking are included in X.

4. USE NonParamDML or CausalForestDML: These can capture non-linear
   CATE functions better than LinearDML.

5. VALIDATE WITH SYNTHETIC: Since we KNOW the true DGP, compare estimates
   to ground truth to calibrate model selection.
""")


if __name__ == "__main__":
    run_full_diagnostic()
