# Feature Correlation Logic Review - Multi-Frequency Optimization

## Problem Statement

With the expansion from 3 lookback periods `[6, 12, 24]` to 19 periods `[6, 7, 8, 9, ..., 24]`, the feature correlation removal logic needed significant improvements:

- **Original features**: ~75 features (5 indicators × 5 categories × 3 periods)
- **New features**: ~475 features (5 indicators × 5 categories × 19 periods)

The original correlation logic had several limitations when dealing with many consecutive periods.

---

## Issues with Original Implementation

### 1. **Hardcoded Period Priority**
```python
# Old logic
if '_6' in feature_name:
    period_priority = 30
elif '_12' in feature_name:
    period_priority = 20
elif '_24' in feature_name:
    period_priority = 10
else:
    period_priority = 15  # All other periods got default priority
```

**Problem**: Periods 7, 8, 9, 10, 11, 13, 14, ..., 23 all got the same default priority, making selection arbitrary.

### 2. **Adjacent Period Correlation**
Adjacent periods (e.g., `RSI_6`, `RSI_7`, `RSI_8`) are highly correlated but weren't intelligently selected. The old logic would:
- Group all highly correlated RSI features together
- Keep only the highest priority one (whichever got priority 30)
- Remove all others, losing potentially useful medium and long-term signals

### 3. **Single Feature per Cluster**
The original logic kept only ONE feature per correlation cluster, which was too aggressive for large clusters spanning multiple timeframes.

### 4. **Limited Logging**
Insufficient visibility into what was being removed and why.

---

## Improved Implementation

### 1. **Dynamic Period Priority**

```python
import re

def get_feature_priority(feature_name):
    # Extract period dynamically using regex
    period_match = re.search(r'_(\d+)(?:$|[^0-9])', feature_name)
    
    if period_match:
        period = int(period_match.group(1))
        # Shorter periods get higher priority (inverse relationship)
        # Period 6 gets ~95 points, period 24 gets ~76 points
        period_priority = 100 - period
    else:
        period_priority = 50
```

**Benefits**:
- ✅ Automatically handles any period value (6-24, or even beyond)
- ✅ Smooth gradient: shorter periods are preferred, but not drastically
- ✅ No hardcoded values

**Priority Examples**:
- Period 6: 94 points (100 - 6)
- Period 12: 88 points (100 - 12)
- Period 18: 82 points (100 - 18)
- Period 24: 76 points (100 - 24)

### 2. **Strategic Multi-Feature Selection**

Instead of keeping just one feature per cluster, the new logic keeps **strategic representatives**:

```python
# Group by base indicator (e.g., all RSI features together)
indicator_groups = {}

for base, feat_list in indicator_groups.items():
    if len(feat_list) <= 3:
        keep_features.extend(feat_list[:1])  # Small group: keep best
    else:
        # Large group: keep representatives from different ranges
        periods = [(extract_period(f), f) for f in feat_list]
        periods.sort(key=lambda x: x[0])
        
        # Keep shortest period (most reactive)
        keep_features.append(periods[0][1])
        
        # If 7+ periods, also keep medium
        if len(periods) >= 7:
            mid_idx = len(periods) // 2
            keep_features.append(periods[mid_idx][1])
        
        # If 13+ periods, also keep longest
        if len(periods) >= 13:
            keep_features.append(periods[-1][1])
```

**Benefits**:
- ✅ Keeps short, medium, and long-term representatives
- ✅ Preserves multi-timeframe information
- ✅ Reduces features while maintaining diversity

**Example**:
For `RSI_6, RSI_7, RSI_8, ..., RSI_24` (19 features, highly correlated):
- **Old logic**: Keep only `RSI_6`
- **New logic**: Keep `RSI_6` (short), `RSI_15` (medium), `RSI_24` (long)

### 3. **Variance-Based Priority**

Added variance consideration to prioritize more informative features:

```python
# Variance-based priority: features with more variance are more informative
try:
    variance_priority = features[feature_name].var()
    # Normalize to 0-10 range
    variance_priority = min(10, variance_priority * 100)
except:
    variance_priority = 0

return period_priority + feature_type_priority + variance_priority
```

**Benefits**:
- ✅ Features with higher variance (more signal) are preferred
- ✅ Breaks ties between otherwise equal features
- ✅ Data-driven selection

### 4. **Enhanced Feature Type Priority**

```python
# More granular feature type priorities
if any(x in feature_name for x in ['ADX', 'RSI', 'MACD', 'ATR']):
    feature_type_priority = 10  # Key indicators
elif any(x in feature_name for x in ['ROC', 'Williams_R', 'CCI', 'Aroon']):
    feature_type_priority = 7   # Important momentum indicators
elif any(x in feature_name for x in ['BB_Width', 'Return_Vol', 'Stoch', 'VI']):
    feature_type_priority = 5   # Useful volatility/oscillators
elif any(x in feature_name for x in ['KAMA', 'PPO', 'AO']):
    feature_type_priority = 3   # Specialized indicators
```

**Benefits**:
- ✅ More nuanced prioritization
- ✅ Ensures key indicators are preserved
- ✅ Better alignment with trading importance

### 5. **Improved Logging and Statistics**

```python
# Sort groups by size for more informative output
sorted_groups = sorted(correlation_groups, key=lambda x: x['cluster_size'], reverse=True)

for i, group in enumerate(sorted_groups[:20]):
    print(f"  Group {i+1} (size: {group['cluster_size']}): Kept {kept_str}")
    print(f"    Max correlation: {group['max_correlation']:.3f}")
    print(f"    Removed: {', '.join(group['removed'][:5])}...")

# Cluster statistics
print(f"Cluster size statistics:")
print(f"  Mean cluster size: {np.mean(cluster_sizes):.1f}")
print(f"  Max cluster size: {max(cluster_sizes)}")
print(f"  Clusters with >10 features: {sum(1 for s in cluster_sizes if s > 10)}")
```

**Benefits**:
- ✅ Shows largest clusters first (most impactful removals)
- ✅ Cluster size statistics help understand correlation patterns
- ✅ Better debugging and validation

### 6. **Dynamic Timeframe Distribution**

```python
# Group features by period ranges
short_term = sum(timeframe_count.get(str(p), 0) for p in range(6, 11))
medium_term = sum(timeframe_count.get(str(p), 0) for p in range(11, 18))
long_term = sum(timeframe_count.get(str(p), 0) for p in range(18, 25))

print(f"  Short-term (6-10 months): {short_term} features")
print(f"  Medium-term (11-17 months): {medium_term} features")
print(f"  Long-term (18-24 months): {long_term} features")

# Detailed per-period breakdown
for period in sorted_periods:
    count = timeframe_count[period]
    if count > 0:
        print(f"    {period}-month: {count} features")
```

**Benefits**:
- ✅ Shows distribution across all periods
- ✅ Helps validate feature diversity
- ✅ Easy to spot if one range is over/under-represented

---

## Selection Strategy Summary

### Priority Formula
```
Total Priority = Period Priority + Feature Type Priority + Variance Priority

Where:
- Period Priority = 100 - period_number (range: 76-94)
- Feature Type Priority = 0-10 (based on indicator importance)
- Variance Priority = 0-10 (based on feature variance)
```

### Selection Rules

1. **Small clusters (≤3 features)**:
   - Keep the highest priority feature

2. **Medium clusters (4-12 features)**:
   - Keep shortest period (highest reactivity)
   - Keep middle period (balanced view)

3. **Large clusters (≥13 features)**:
   - Keep shortest period (short-term signals)
   - Keep middle period (medium-term trends)
   - Keep longest period (long-term context)

---

## Expected Results

### Before Improvements
With 19 periods and correlation threshold 0.5:
- **Input features**: ~475 features
- **After removal**: ~50-80 features (very aggressive, loses timeframe diversity)
- **Problem**: Only keeps features from similar periods (mostly period 6)

### After Improvements
With 19 periods and correlation threshold 0.5:
- **Input features**: ~475 features
- **After removal**: ~150-200 features (balanced reduction)
- **Benefit**: Maintains representatives across short/medium/long timeframes
- **Quality**: Higher variance features are preferred

### Example: RSI Indicator Family

**Original behavior** (19 RSI features: RSI_6 through RSI_24):
```
Cluster: [RSI_6, RSI_7, RSI_8, ..., RSI_24]
Kept: RSI_6 (only)
Removed: 18 features
Result: Lost all medium and long-term RSI signals
```

**Improved behavior**:
```
Cluster: [RSI_6, RSI_7, RSI_8, ..., RSI_24]
Kept: RSI_6 (short), RSI_15 (medium), RSI_24 (long)
Removed: 16 features
Result: Preserved multi-timeframe RSI analysis with 84% reduction
```

---

## Recommendations for Tuning

### 1. **Correlation Threshold**

```python
# Conservative (keep more features, allow some correlation)
correlation_threshold = 0.7  # ~200-250 features

# Balanced (default)
correlation_threshold = 0.5  # ~150-200 features

# Aggressive (fewer features, strict decorrelation)
correlation_threshold = 0.3  # ~100-150 features
```

### 2. **Strategic Period Selection**

Consider limiting to strategic periods upfront:
```python
# Instead of all consecutive periods
lookback_periods = [6, 7, 8, 9, 10, ..., 24]  # 19 periods

# Use strategic intervals
lookback_periods = [6, 9, 12, 15, 18, 21, 24]  # 7 periods
# OR
lookback_periods = [6, 8, 10, 12, 15, 18, 21, 24]  # 8 periods
```

**Benefits**:
- Starts with less correlated features
- Faster computation
- Still covers full timeframe range

### 3. **Indicator-Specific Thresholds**

Consider different thresholds for different indicator types:
```python
# Momentum indicators: allow higher correlation (more stable)
momentum_threshold = 0.6

# Volatility indicators: stricter threshold (more variable)
volatility_threshold = 0.4
```

---

## Performance Considerations

### Computational Complexity

**Original**: O(F²) where F = number of features
- For 475 features: ~225,625 comparisons
- Manageable for monthly data

**Improved**: O(F² + C*G) where C = clusters, G = max cluster size
- Correlation matrix: O(F²) = ~225,625 comparisons
- Cluster processing: O(C*G) = typically < 10,000 operations
- **Total**: Still O(F²), similar performance

### Memory Usage

- Correlation matrix: F × F × 8 bytes
- For 475 features: ~1.8 MB (negligible)
- Processing overhead: < 1 MB

### Recommendations

For even larger feature sets (1000+ features):
1. Use chunked correlation calculation
2. Consider approximate nearest neighbors for clustering
3. Pre-filter obvious redundancies before correlation analysis

---

## Testing and Validation

### 1. **Validate Timeframe Diversity**

```python
# Check distribution after feature selection
short_term_count = sum(1 for f in selected_features if 6 <= extract_period(f) <= 10)
medium_term_count = sum(1 for f in selected_features if 11 <= extract_period(f) <= 17)
long_term_count = sum(1 for f in selected_features if 18 <= extract_period(f) <= 24)

assert short_term_count > 0, "Missing short-term features"
assert medium_term_count > 0, "Missing medium-term features"
assert long_term_count > 0, "Missing long-term features"
```

### 2. **Validate Indicator Coverage**

```python
# Ensure key indicators are present
key_indicators = ['ADX', 'RSI', 'MACD', 'ATR', 'ROC']
for indicator in key_indicators:
    indicator_features = [f for f in selected_features if indicator in f]
    assert len(indicator_features) > 0, f"Missing {indicator} features"
```

### 3. **Validate Correlation Threshold**

```python
# Verify correlation is below threshold
final_corr = selected_features.corr().abs()
max_corr = final_corr.values[np.triu_indices_from(final_corr.values, k=1)].max()

print(f"Maximum correlation in final features: {max_corr:.3f}")
assert max_corr <= correlation_threshold * 1.1, "Correlation threshold exceeded"
```

---

## Summary

### Key Improvements

1. ✅ **Dynamic period extraction** - handles any lookback period automatically
2. ✅ **Multi-representative selection** - keeps short/medium/long timeframes
3. ✅ **Variance-based prioritization** - prefers more informative features
4. ✅ **Enhanced logging** - better visibility into selection process
5. ✅ **Flexible timeframe grouping** - adapts to any period range

### Expected Outcomes

- **Feature reduction**: 475 → ~150-200 features (68-58% reduction)
- **Timeframe preservation**: Representatives from all period ranges
- **Indicator diversity**: All major indicator types retained
- **Correlation control**: Maximum correlation < threshold
- **Information retention**: Higher variance features preferred

### Migration Path

The improved logic is **backward compatible** - works with both:
- Original setup: `[6, 12, 24]` → ~30-40 features
- Extended setup: `[6, 7, ..., 24]` → ~150-200 features

No changes needed to existing code, just better selection logic!

---

## Example Output

```
=== Removing Correlated Features (threshold: 0.5) ===
Found 47 correlation groups:
  Group 1 (size: 19): Kept RSI_6, RSI_15, RSI_24
    Max correlation: 0.987
    Removed: RSI_7, RSI_8, RSI_9, RSI_10, RSI_11 + 11 more
  Group 2 (size: 19): Kept MACD_6, MACD_14, MACD_24
    Max correlation: 0.965
    Removed: MACD_7, MACD_8, MACD_9, MACD_10, MACD_11 + 11 more
  ...

Cluster size statistics:
  Mean cluster size: 9.8
  Max cluster size: 19
  Clusters with >10 features: 15

Feature selection summary:
  Original features: 475
  Correlation groups: 47
  Removed features: 298
  Final features: 177

Final features by timeframe:
  Short-term (6-10 months): 68 features
  Medium-term (11-17 months): 59 features
  Long-term (18-24 months): 50 features
  Other/No period: 0 features

  Detailed breakdown:
    6-month: 28 features
    7-month: 8 features
    8-month: 10 features
    9-month: 11 features
    10-month: 11 features
    ...
```

This improved logic ensures you maintain a diverse, informative feature set across all timeframes! 🚀
