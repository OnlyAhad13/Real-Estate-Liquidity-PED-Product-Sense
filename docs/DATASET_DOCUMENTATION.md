# Synthetic Marketplace Dataset Documentation

## Overview

This document provides comprehensive documentation for the synthetic Airbnb-style marketplace dataset generated for **Causal Inference** analysis, specifically designed to estimate **Price Elasticity of Demand (PED)**.

**Dataset:** `marketplace_data.csv`  
**Records:** 10,000 property listings  
**Purpose:** Simulate realistic marketplace data with known confounding structures for causal inference method validation

---

## Dataset Schema

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `listing_id` | Integer | 1 - 10,000 | Unique property identifier |
| `room_type` | Categorical | Shared, Private, Entire | Type of accommodation |
| `property_tier` | Categorical | Luxury, Economy | Derived property classification |
| `location_score` | Float | 0.0 - 10.0 | Location quality rating |
| `capacity` | Integer | 1 - 12 | Maximum guest capacity |
| `host_rating` | Float | 3.0 - 5.0 | Host performance rating |
| `seasonality_index` | Float | 0.5 - 2.0 | Time-varying demand multiplier |
| `historical_price` | Float | 20 - 500 | Listed nightly price (USD) |
| `booking_probability` | Float | 0.0 - 1.0 | True booking probability |
| `is_booked` | Binary | 0, 1 | Observed booking outcome |
| `competitor_density` | Integer | 0 - ~50 | Nearby competing listings |
| `host_experience_days` | Integer | 1 - ~2000+ | Days since host joined |
| `platform_fee_rate` | Float | 0.10 - 0.18 | Platform service fee rate |
| `cleaning_cost` | Float | 10 - 100 | Cleaning fee (USD) |

---

## Feature Definitions & Distribution Rationale

### 1. `location_score` (0-10)

**Definition:** A composite score representing the desirability of a property's location, incorporating factors like proximity to attractions, transit access, safety, and neighborhood quality.

**Distribution:** `Beta(2, 2) × 10`

**Rationale:**
- **Why Beta(2, 2)?** This symmetric, bell-shaped distribution creates realistic clustering around mid-range values (4-6), reflecting that most properties aren't in the absolute best or worst locations.
- **Real-world parallel:** Airbnb's review system and internal ranking algorithms typically produce similar distributions—few properties score at extremes, most cluster in the middle.
- **Why 0-10?** Industry-standard rating scale that's intuitive for users and easily interpretable in regression coefficients.

```
Distribution Shape:
     ▃▅▇█▇▅▃
  0  2  4  6  8  10
       ↑ Peak ~5
```

---

### 2. `room_type` (Categorical)

**Definition:** The type of accommodation being offered.

**Categories & Probabilities:**
| Type | Probability | Description |
|------|-------------|-------------|
| Shared | 10% | Shared space (hostel-style) |
| Private | 25% | Private room in shared home |
| Entire | 65% | Entire home/apartment |

**Rationale:**
- **Based on Airbnb data:** Real Airbnb listings show ~60-70% are entire homes, ~20-25% private rooms, and ~10% shared spaces (source: Inside Airbnb datasets).
- **Economic significance:** Room type is a major determinant of both price and demand, making it essential for proper confounding control.
- **Causal structure:** Entire homes command premium prices but also attract different demand patterns than shared rooms.

---

### 3. `capacity` (1-12 guests)

**Definition:** Maximum number of guests the property can accommodate.

**Distribution:** `Poisson(λ=3) + 1`, clipped to [1, 12]

**Rationale:**
- **Why Poisson?** Guest capacity follows a count distribution with natural positive skew—most properties accommodate 2-4 guests, with fewer large properties.
- **Why λ=3?** Creates a realistic mode of 3-4 guests, matching typical couples/small family stays.
- **Why +1 shift?** Ensures minimum capacity of 1 (no zero-capacity properties).
- **Why cap at 12?** Airbnb typically limits group bookings; properties >12 guests are rare and often subject to special policies.

```
Expected Distribution:
Capacity:  1   2   3   4   5   6   7+
Frequency: 5% 15% 25% 20% 15% 10% 10%
```

---

### 4. `host_rating` (3.0-5.0)

**Definition:** Average rating given to the host based on guest reviews.

**Distribution:** `3.0 + Beta(5, 2) × 2`

**Rationale:**
- **Why Beta(5, 2)?** This left-skewed distribution creates realistic bunching near the upper bound (4.5-5.0), reflecting the well-documented rating inflation on review platforms.
- **Why start at 3.0?** Properties with ratings below 3.0 rarely survive on the platform—they either improve or are delisted. This survivorship bias is realistic.
- **Why cap at 5.0?** Standard 5-star rating system.
- **Real-world evidence:** Studies show Airbnb average ratings cluster around 4.7-4.8, with very few below 4.0.

```
Distribution Shape:
         ▃▅▇█
  3.0  3.5  4.0  4.5  5.0
              ↑ Peak ~4.5
```

---

### 5. `seasonality_index` (0.5-2.0)

**Definition:** A time-varying demand multiplier representing seasonal fluctuations, local events, holidays, and market conditions.

**Distribution:** `0.5 + Beta(2, 2) × 1.5`

**Rationale:**
- **Why this is the KEY CONFOUNDER:** Seasonality affects both price (hosts raise prices during peak times) and demand (more travelers during holidays/summer). This creates the **endogeneity problem** central to causal inference.
- **Why 0.5-2.0?** Represents demand ranging from 50% of baseline (off-season) to 200% (peak season/major events).
- **Why Beta(2, 2)?** Most observations are in "normal" periods (index ~1.0-1.25), with fewer extreme low/high seasons.
- **Temporal interpretation:** Though cross-sectional, think of each listing as observed at a random point in time.

| Index Value | Season Interpretation |
|-------------|----------------------|
| 0.5 - 0.8 | Off-season (Jan-Feb, Nov) |
| 0.8 - 1.2 | Normal demand |
| 1.2 - 1.5 | High season (Summer, Spring Break) |
| 1.5 - 2.0 | Peak events (NYE, major festivals) |

---

### 6. `property_tier` (Derived: Luxury/Economy)

**Definition:** A classification of properties into market segments based on observable characteristics.

**Derivation Formula:**
```
luxury_score = 0.4 × (location_score/10) + 
               0.3 × ((host_rating - 3)/2) + 
               0.3 × room_type_premium

property_tier = "Luxury" if luxury_score > 0.6 else "Economy"
```

**Rationale:**
- **Why create this?** Enables modeling of **heterogeneous treatment effects**—the key insight that price sensitivity varies by market segment.
- **Economic theory:** Luxury goods have lower price elasticity because buyers are less price-sensitive (Veblen effects, quality signaling).
- **Weights chosen:** Location (40%), Rating (30%), Room Type (30%) reflect real pricing determinants.

---

### 7. `historical_price` (USD, $20-$500)

**Definition:** The nightly price set by the host, reflecting revenue management decisions.

**Generation Formula (Confounded Pricing):**
```
price = base_price ($50) +
        location_premium (0-$40) +
        room_premium ($0/$25/$60) +
        capacity_premium ($8/guest) +
        rating_premium (0-$30) +
        seasonality_premium (0-$60) +  ← CONFOUNDING
        noise (σ=$15)
```

**Critical Design Choice - Endogeneity:**
The **seasonality premium** creates **positive correlation between price and demand**, simulating real-world revenue management where hosts raise prices when demand is high. This is the **confounding we need to solve** in causal inference.

**Rationale for Bounds:**
- **$20 floor:** Below this, hosting isn't economically viable (cleaning, utilities, time costs).
- **$500 ceiling:** Represents luxury urban properties; higher prices exist but are rare.
- **Premium structure:** Based on Airbnb pricing studies showing location, room type, and capacity as primary price drivers.

---

### 8. `booking_probability` & `is_booked`

**Definition:** The true probability of booking and the realized binary outcome.

**True Data Generating Process (DGP):**
```
logit(P(booked)) = 1.5 +
                   β_price × price +        ← TRUE CAUSAL EFFECT
                   0.8 × seasonality +
                   0.15 × location_score +
                   room_type_effect +
                   0.05 × capacity +
                   0.4 × (rating - 3) +
                   noise(σ=0.3)
```

**Heterogeneous Price Coefficients:**
| Property Tier | β_price | Interpretation |
|---------------|---------|----------------|
| **Luxury** | -0.008 | $100 increase → ~45% odds reduction |
| **Economy** | -0.015 | $100 increase → ~78% odds reduction |

**Rationale:**
- **Negative price coefficient:** Economic law of demand—higher prices reduce purchase probability.
- **Heterogeneous effects:** Luxury buyers are less price-sensitive than budget-conscious travelers.
- **Logistic model:** Appropriate for binary outcomes with bounded probability.

---

## Instrumental Variables (For Causal Inference)

These variables are designed to satisfy the instrumental variable assumptions for **2SLS regression** or similar methods.

### 9. `competitor_density`

**Distribution:** `Poisson(λ=15)`

**Instrument Logic:**
- **Relevance:** More competitors → hosts lower prices (competitive pressure) ✓
- **Exclusion:** Number of competitors doesn't directly affect an individual booking decision (conditional on price) ✓
- **Exogeneity:** Competitor count is determined by market factors, not individual listing's demand ✓

---

### 10. `host_experience_days`

**Distribution:** `Exponential(λ=1/365)` (mean ~1 year)

**Instrument Logic:**
- **Relevance:** Experienced hosts may price more optimally (higher or lower strategically)
- **Exclusion:** Guest doesn't know/care how long host has been on platform
- **Potential weakness:** Experience might correlate with service quality (work to validate)

---

### 11. `platform_fee_rate`

**Distribution:** `Uniform(0.10, 0.18)`

**Instrument Logic:**
- **Relevance:** Higher platform fees → hosts raise prices to maintain margins ✓
- **Exclusion:** Guests don't observe platform fee rates ✓
- **Exogeneity:** Fee variations are random (A/B testing, different host tiers) ✓
- **This is the strongest instrument** in the dataset.

---

### 12. `cleaning_cost`

**Distribution:** `15 + 5 × capacity + Normal(0, 5)`, clipped to [10, 100]

**Instrument Logic:**
- **Relevance:** Higher cleaning costs → higher listing prices (cost pass-through)
- **Exclusion:** Cleaning cost is a supply-side factor, not demand-side
- **Capacity linkage:** Larger properties cost more to clean (realistic)

---

## Causal Structure Summary

```
                    ┌─────────────────┐
                    │  Seasonality    │
                    │    Index        │
                    └────────┬────────┘
                             │
              CONFOUNDING    │    CONFOUNDING
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              │              ▼
     ┌────────────────┐      │      ┌────────────────┐
     │  Historical    │      │      │   Booking      │
     │    Price       │──────┴─────→│   Outcome      │
     └────────────────┘              └────────────────┘
              ▲              TRUE CAUSAL
              │              EFFECT (-)
     ┌────────────────┐
     │  Instruments   │
     │  (Fee, Cost,   │
     │   Competitors) │
     └────────────────┘
```

**Key Insight:** Naive regression of `is_booked` on `historical_price` will show **biased** (likely positive) coefficient due to confounding. Proper causal methods (IV, matching, Double ML) should recover the **true negative effect**.

---

## Validation Checks

When you run the generation script, it performs automatic validation:

| Check | Expected | Purpose |
|-------|----------|---------|
| Price-Seasonality Corr > 0.3 | ✓ ~0.41 | Confirms confounding exists |
| Price-Booking Corr (observed) > 0 | ✓ ~0.38 | Confirms endogeneity bias |
| Economy more price-sensitive | ✓ | Confirms heterogeneous effects |

---

## Usage Notes

### For Causal Inference Methods:
1. **Naive OLS** on price → booking will be biased upward (positive instead of negative)
2. **IV Regression** using `platform_fee_rate` or `cleaning_cost` should recover negative effect
3. **Propensity Score Matching** on `seasonality_index` + other confounders
4. **Double/Debiased ML** using all features with proper sample splitting

### Known Ground Truth:
- **True ATE (Average Treatment Effect):** ~-0.012 per $1 price increase
- **Luxury CATE:** -0.008 per $1
- **Economy CATE:** -0.015 per $1

---

## References & Inspiration

- Inside Airbnb Project (distribution benchmarks)
- Zervas, G., Proserpio, D., & Byers, J. (2017). "The Rise of the Sharing Economy"
- Fradkin, A. (2017). "Search, Matching, and the Role of Digital Marketplaces"
- Angrist, J. & Pischke, J.S. (2009). "Mostly Harmless Econometrics" (IV methodology)

---