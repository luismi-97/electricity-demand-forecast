## Results

### Final test (locked period, evaluated once)

- MAE of **Blend (LightGBM + NN)**: 1,299 MWh vs. 1,980 MWh for the official TSO forecast (34.4 % lower error); paired bootstrap over days: difference -681 MWh (95 % CI -759 to -603), lower error on 75 % of days
- MAE of **Blend (LightGBM + NN)**: 1,299 MWh vs. 2,697 MWh for the weekly naive baseline (51.9 % lower error); paired bootstrap over days: difference -1,399 MWh (95 % CI -1,570 to -1,230), lower error on 84 % of days

Point forecasts (MAE, RMSE and bias in MWh; MAPE in %; MASE scaled by the weekly naive):

| Model | n | MAE | RMSE | MAPE_% | bias | MASE |
|---|---|---|---|---|---|---|
| Blend (LightGBM + NN) | 16,224 | 1,299 | 1,682 | 2.45 | -278 | 0.54 |
| Neural network | 16,224 | 1,346 | 1,769 | 2.54 | -300 | 0.56 |
| LightGBM | 16,224 | 1,372 | 1,783 | 2.59 | -256 | 0.57 |
| LightGBM (quantile, q50) | 16,224 | 1,403 | 1,827 | 2.65 | -263 | 0.58 |
| Official TSO forecast | 16,224 | 1,980 | 2,547 | 3.79 | +316 | 0.82 |
| Naive (same hour last week) | 16,224 | 2,697 | 3,986 | 5.13 | -95 | 1.11 |

90 % prediction intervals (nominal coverage 0.90):

| Interval | n | nominal | coverage | mean_width | width_%_of_load |
|---|---|---|---|---|---|
| raw quantile | 16,224 | 0.90 | 0.672 | 3,818 | 7.3 |
| conformalised | 15,480 | 0.90 | 0.901 | 6,421 | 12.3 |

### Validation 2023-2024 (used for model development)

- MAE of **Blend (LightGBM + NN)**: 1,217 MWh vs. 1,953 MWh for the official TSO forecast (37.7 % lower error); paired bootstrap over days: difference -736 MWh (95 % CI -825 to -650), lower error on 74 % of days
- MAE of **Blend (LightGBM + NN)**: 1,217 MWh vs. 2,498 MWh for the weekly naive baseline (51.3 % lower error); paired bootstrap over days: difference -1,281 MWh (95 % CI -1,460 to -1,110), lower error on 79 % of days

Point forecasts (MAE, RMSE and bias in MWh; MAPE in %; MASE scaled by the weekly naive):

| Model | n | MAE | RMSE | MAPE_% | bias | MASE |
|---|---|---|---|---|---|---|
| Blend (LightGBM + NN) | 17,544 | 1,217 | 1,613 | 2.34 | +210 | 0.50 |
| LightGBM | 17,544 | 1,264 | 1,686 | 2.43 | +161 | 0.52 |
| LightGBM (quantile, q50) | 17,544 | 1,295 | 1,737 | 2.50 | +126 | 0.54 |
| Neural network | 17,544 | 1,314 | 1,752 | 2.52 | +258 | 0.54 |
| Official TSO forecast | 17,544 | 1,953 | 2,472 | 3.76 | -115 | 0.81 |
| Naive (same hour last week) | 17,544 | 2,498 | 3,939 | 4.89 | -15 | 1.03 |

90 % prediction intervals (nominal coverage 0.90):

| Interval | n | nominal | coverage | mean_width | width_%_of_load |
|---|---|---|---|---|---|
| raw quantile | 17,544 | 0.90 | 0.659 | 3,599 | 6.9 |
| conformalised | 16,800 | 0.90 | 0.894 | 5,813 | 11.3 |

> Caveats: the models see the **observed** weather of the target day (the official forecast was made with weather forecasts) and use load up to 12:00 on D-1, which may differ from the TSOs' information cutoff. Read the comparison as 'under this protocol', not as a like-for-like contest.
