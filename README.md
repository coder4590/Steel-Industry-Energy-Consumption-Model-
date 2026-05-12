# ⚡ The Invisible Thief — Steel Industry Energy Consumption Prediction

> The factory is running. Machines are humming. Somewhere in the control room, a screen shows numbers. A lot of numbers. But nobody is looking at them.

> That's the problem. The steel plant is bleeding energy — not from leaks or broken pipes, but from inefficiency. Machines running when they shouldn't. Maximum load when medium would do. Reactive power spiking without anyone noticing.

> What if a machine could watch those numbers? Not just watch — understand. Learn the rhythm of the factory. Know when the plant is running light, medium, or at full blast. And flag it, before the electricity bill arrives.

> **That's what this project does.**

---

## 🏭 What This Actually Is

A machine learning system that predicts the operational load type of a steel plant — Light Load, Medium Load, or Maximum Load — using real-time electrical measurements and time features.

| Input | Output |
|---|---|
| 9 electrical and time features | Load Type: Light, Medium, or Maximum |
| 35,040 rows of real sensor data | 75.2% accuracy on 3-class prediction |
| Time-series aware splitting | Honest evaluation, no future leakage |

---

## 📊 The Data

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | 35,040 (15-minute intervals) |
| Original Features | 9 (7 electrical + 2 time indicators) |
| Target | Load Type: Light Load, Medium Load, Maximum Load |
| Dropped Features | 2 (Leading Reactive Power, Leading Power Factor — always near constant) |
| Time Dependency | Strong daily and weekly cycles. Cannot shuffle. |

---

## 💡 The Core Insight — Medium and Maximum Load Overlap

| Finding | Impact |
|---|---|
| Medium and Maximum Load overlap significantly | No algorithm can fully separate them |
| Usage patterns are similar across both classes | Even a human operator would misclassify ~25% |
| Reactive power ranges overlap | The data itself contains ambiguity |
| This is not a model failure | It's a data ceiling |

---

## 🔍 Data Visualization — Nine Plots, Nine Stories

Every plot was designed to answer one specific business question about the steel plant's energy behavior.

---

### Plot 1: Average Usage (kWh) by Hour & Load Type

<img width="1800" height="750" alt="image" src="https://github.com/user-attachments/assets/783117cc-3afa-402a-9abe-60db0569af4b" />

| Type | Line Plot |
|---|---|
| **Question** | How does energy usage vary throughout the day for each load type? |
| **Finding** | Light Load dominates at night (0-5 AM). Medium and Maximum Load cluster during working hours (8 AM-5 PM) but overlap heavily — usage alone cannot separate them. |
| **Business Insight** | The factory has a clear daily rhythm. But if you only look at usage, you cannot tell if the plant is running medium or maximum production. |

---

### Plot 2: Lagging Reactive Power — Weekday vs Weekend

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/259f4bd4-e64e-4897-ad81-908d062218e8" />


| Type | Side-by-Side Boxplot |
|---|---|
| **Question** | Is wasted energy different on weekdays versus weekends for the same load type? |
| **Finding** | Maximum Load on weekdays shows much higher reactive power than weekends. The same load type behaves differently depending on the day. |
| **Business Insight** | Even when running at "maximum," the plant wastes less energy on weekends. Fewer heavy inductive machines are active. WeekStatus is a strong predictor. |

---

### Plot 4: CO2 Emissions Throughout the Day by Load Type

<img width="1800" height="900" alt="image" src="https://github.com/user-attachments/assets/197036e8-cbc7-4270-be06-e6c8614b7f80" />

| Type | Line Plot (Pivot) |
|---|---|
| **Question** | Does CO2 follow a daily rhythm and separate load types? |
| **Finding** | The CO2 plot is nearly identical to the Usage plot. CO2 is calculated directly from electricity consumption using a fixed carbon factor. |
| **Business Insight** | CO2 is a duplicate in disguise. It adds no new information. Keep Usage, drop CO2, or keep both — but know they carry the same signal. |

---

### Plot 5: Lagging Power Factor Density by Load Type

<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/f2a0b076-1ed3-4419-8e1c-05c0e86cb7e6" />

| Type | KDE Density Plot |
|---|---|
| **Question** | What efficiency ranges do different load types operate in? |
| **Finding** | Maximum Load operates at lower power factor (60-70 range — poorer efficiency). Light Load runs near 95-100 (near perfect efficiency). Medium Load sits in between. |
| **Business Insight** | Power factor is a genuine signal. The heavier the load, the worse the efficiency. But the curves overlap — it's not a clean separator. |

---

### Plot 6: Maximum Load — Count per Day & Hour

<img width="2100" height="1200" alt="image" src="https://github.com/user-attachments/assets/25d93ef9-2779-4c70-948a-8516ff169a85" />

| Type | Heatmap |
|---|---|
| **Question** | When exactly does peak production happen across the week? |
| **Finding** | Sunday is nearly blank — almost no Maximum Load. Monday through Friday from 8 AM to 17 PM burn bright. Saturday shows lighter activity. |
| **Business Insight** | Day_of_week and hour are critical predictors. The factory rests on Sunday and runs hard on weekdays. This pattern is rock-solid. |

---

### Plot 7: Leading Power Factor Density by Load Type

<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/4a5e8f71-03dc-4d72-955a-6514ed10070d" />

| Type | KDE Density Plot |
|---|---|
| **Question** | Does leading power factor vary across load types? |
| **Finding** | All three curves overlap completely at 100. No separation. No variance. |
| **Business Insight** | Leading power factor is useless. It was dropped alongside leading reactive power. Two features removed with zero signal loss. |

---

### Plot 8: Raw Usage (kWh) — First 1000 Records

<img width="2100" height="750" alt="image" src="https://github.com/user-attachments/assets/558222f4-eba4-40c9-9d7d-c10375bde580" />

| Type | Time-Series Line Plot |
|---|---|
| **Question** | Is there a visible time pattern in the raw data? |
| **Finding** | Clear repeating cycles. Sharp peaks during the day. Deep valleys at night. Regular rhythm across multiple days. |
| **Business Insight** | This data is strongly time-dependent. Shuffling rows would destroy the temporal structure and give fake high accuracy. TimeSeriesSplit is mandatory. |

---

### Plot 9: Usage vs Lagging Reactive Power by Load Type

<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/b63d7be8-0025-44fc-9c64-177246a5baac" />

| Type | Colored Scatter Plot |
|---|---|
| **Question** | Do two strong features together separate the classes? |
| **Finding** | Three distinct clusters are visible. Light Load sits in the bottom-left corner (low usage, low waste). Maximum Load spreads across the top-right. Medium Load occupies the middle band. But boundaries blur — the clusters overlap at the edges. |
| **Business Insight** | The signal is real. The classes are separable but not perfectly. This scatter plot captures the entire story of the dataset: genuine pattern, genuine limitation. |

---

## 🧪 The Model — Three Algorithms, One Ceiling

| Model | Accuracy | Verdict |
|---|---|---|
| Random Forest | 74.2% | Bagging handles noise but can't fix class overlap |
| LightGBM | 75.0% | Leaf-wise growth crashed repeatedly. Heavily constrained. |
| **XGBoost** | **75.2%** | **Winner.** Level-wise growth. Stable. Best for this data size. |

**Why 75.2% is the honest ceiling:** Three fundamentally different algorithms. Three different approaches to learning. All converged within 1% of each other. The data cannot give more.

---

## 📈 Confusion Matrix

| Actual ↓ / Predicted → | Light Load | Medium Load | Maximum Load |
|---|---|---|---|
| **Light Load** | 3,491 (95%) | 12 | 185 |
| **Medium Load** | 19 | 686 (48%) | 711 |
| **Maximum Load** | 37 | 773 | 1,078 (57%) |

| Key Insight |
|---|
| Light Load is easy — 95% recall. The model almost never misses an idle period. |
| Medium Load bleeds into Maximum — only 48% correctly identified. |
| Maximum Load confuses with Medium — 57% recall. The overlap zone. |
| This is not a bug. It's the physical reality of the steel plant. |

---

## 🏗️ Pipeline Architecture

| Stage | What Happened |
|---|---|
| **Data Cleaning** | Missing values checked. Duplicates removed. 12 impossible value checks. Outlier detection with boxplot grid. 2 useless features dropped. |
| **Feature Engineering** | Hour extracted from NSM. Cyclic encoding considered. Interaction features tested. Conclusion: original features already contained all signal. |
| **Time-Aware Split** | Manual 80/20 split using iloc. No shuffling. Past predicts future. |
| **Cross-Validation** | TimeSeriesSplit with 5 folds. Industry standard for time-series. |
| **Preprocessing** | ColumnTransformer: StandardScaler for numeric, OneHotEncoder for WeekStatus and Day_of_week. |
| **Model** | XGBoost classifier with 30 iterations of RandomizedSearchCV across 9 hyperparameters. |
| **Evaluation** | Accuracy, precision, recall, F1-score per class. Confusion matrix. Feature importance. CV vs test comparison. |

---

## 📈 Key Metrics

| Metric | Value | Meaning |
|---|---|---|
| Test Accuracy | 75.2% | 2.25x better than random (33%) |
| CV Accuracy | 88.6% | Higher than test — expected with time-series |
| Light Load Recall | 95% | Almost never misses idle |
| Medium Load Recall | 48% | The overlap zone |
| Maximum Load Recall | 57% | Better than Medium, still bleeds |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn (9 distinct plot types) |
| ML Models | XGBoost, LightGBM, RandomForest |
| Pipeline | ColumnTransformer, Pipeline, StandardScaler, OneHotEncoder |
| Tuning | RandomizedSearchCV, TimeSeriesSplit |
| Evaluation | accuracy_score, classification_report, confusion_matrix |

---

## 📝 Lessons Learned

| # | Lesson |
|---|---|
| 1 | Some classes physically overlap. No algorithm can separate what nature made similar. Accepting a data ceiling is not failure — it's honesty. |
| 2 | Time-series data demands respect. Shuffling would have given fake 90%+ accuracy. TimeSeriesSplit told the truth. |
| 3 | Feature engineering isn't magic. On well-measured industrial data, the original sensors capture everything. |
| 4 | Three algorithms, one answer = the ceiling. Stop tuning. Ship it. |
| 5 | Industrial data is underrated. Factories have real problems and real data — and nobody is solving them. |

---

## 👤 Author

A machine learning engineer who believes that saving energy in a steel plant is just as important as predicting disease in a hospital. Both save lives. One saves the planet too.

---

*"The machines are talking. Someone just needs to listen."*
