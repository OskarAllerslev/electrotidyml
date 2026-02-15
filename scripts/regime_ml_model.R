

# load data ind ----

sp500 <- get_smart_data(
  ticker = "^GSPC"
)
vix <- get_smart_data(
  ticker = "^VIX"
)
rate10y <- get_smart_data("^TNX")

xly <- get_smart_data("XLY") # Consumer Discretionary (Luksus/Cyklisk)
xlp <- get_smart_data("XLP") # Consumer Staples (Nødvendighed/Defensiv)

hyg <- get_smart_data("HYG") # High Yield Corporate Bonds (Junk)
ief <- get_smart_data("IEF") # 7-10 Year Treasury (Sikker havn)

rate_short <- get_smart_data("^IRX")


## merge data -----
raw_data <- merge(
  quantmod::Ad(sp500),
  quantmod::Ad(vix),
  quantmod::Ad(rate10y),
  quantmod::Ad(rate_short),
  quantmod::Ad(xly),
  quantmod::Ad(xlp),
  quantmod::Ad(hyg),
  quantmod::Ad(ief)
)
colnames(raw_data) <- c("sp500", "vix", "rate10y", "rate_short", "xly", "xlp", "hyg", "ief")
data_tbl <- raw_data %>%
  timetk::tk_tbl(preserve_index = TRUE, rename_index = "date") %>%
  dplyr::filter(complete.cases(.))



# feature engineering ----
data_final <- data_tbl %>%
  dplyr::mutate(
    # 50-dages glidende gennemsnit til trend-identifikation
    sma_50 = TTR::SMA(sp500, n = 50),
    yield_curve = rate10y - rate_short,
    credit_ratio = hyg / ief,
    risk_appetite = xly / xlp,
    mom_20 = sp500 / dplyr::lag(sp500, 20) - 1,
    future_price_20 = dplyr::lead(sp500, 20),
    future_sma_20   = dplyr::lead(sma_50, 20),
    target_regime = ifelse(future_price_20 < future_sma_20, "bear", "bull"),
    target_regime = factor(target_regime, levels = c("bear", "bull"))
  ) %>%
  # Vi fjerner NA fra SMA (de første 50 rækker) og Target (de sidste 20 rækker)
  dplyr::filter(!is.na(target_regime), !is.na(sma_50), !is.na(mom_20))

# Tjek den nye fordeling
table(data_final$target_regime)
### plot over regime ----
p <- ggplot2::ggplot(
  data = data_final, 
  mapping = ggplot2::aes(
    x = date, 
    y = sp500, 
    color = target_regime,
    group = 1
  )
) + 
  ggplot2::geom_line(linewidth = 1)  



# recipe ----

# bliver vist for domineret
regime_recipe <- recipes::recipe(target_regime ~ ., data = data_final) |> 
  recipes::update_role(date, new_role = "ID") |> 
  recipes::step_rm(future_price_20, future_sma_20, sma_50) |> 
  recipes::step_window(
    credit_ratio, 
    size = 21, 
    statistic = "mean", 
    names = "credit_ratio_monthly_avg"
  ) |> 
  recipes::step_mutate(
    vix_level = vix / 100, 
    yield_slope = yield_curve
  ) |> 
  recipes::step_date(
    date, 
    features = c("dow", "month"), 
    label = FALSE
  ) |> 
  recipes::step_lag(
    recipes::all_predictors(), 
    -recipes::all_outcomes(), 
    lag = 1
  ) |> 
  recipes::step_naomit(recipes::all_predictors()) |> 
  recipes::step_zv(recipes::all_predictors()) |> 
  recipes::step_normalize(recipes::all_numeric_predictors())

regime_recipe <- recipes::recipe(target_regime ~ ., data = data_final) |> 
  recipes::update_role(date, new_role = "ID") |> 
  recipes::step_rm(
    future_price_20, future_sma_20, sma_50,
    rate10y, rate_short, yield_curve, 
    sp500, xly, xlp, hyg, ief
  ) |> 
  recipes::step_mutate(
    credit_trend = credit_ratio / dplyr::lag(credit_ratio, 10) - 1,
    risk_trend   = risk_appetite / dplyr::lag(risk_appetite, 10) - 1,
    vix_trend    = vix / dplyr::lag(vix, 10) - 1
  ) |> 
  recipes::step_rm(credit_ratio, risk_appetite, vix) |>
  recipes::step_date(date, features = c("dow", "month"), label = FALSE) |> 
  recipes::step_lag(recipes::all_predictors(), -recipes::all_outcomes(), lag = 1) |> 
  recipes::step_naomit(recipes::all_predictors()) |> 
  recipes::step_zv(recipes::all_predictors()) |> 
  recipes::step_normalize(recipes::all_numeric_predictors())


# Nu burde denne køre uden fejl
check_data <- recipes::prep(regime_recipe) |> 
  recipes::juice()

dplyr::glimpse(check_data)









# fast cv ----
set.seed(123)
data_split <- rsample::initial_time_split(data_final, prop = 0.8)
train_data <- rsample::training(data_split)
test_data  <- rsample::testing(data_split)

regime_folds_light <- rsample::rolling_origin(
  data       = train_data,
  initial    = 1250, 
  assess     = 250,   
  skip       = 500,     
  cumulative = TRUE
)

# specificere model ----
xgb_spec_light <- parsnip::boost_tree(
  trees = 1000, 
  tree_depth = 5, 
  learn_rate = 0.02, 
  mtry = 3,           # VIKTIGT: Sæt til et heltal (f.eks. 5 ud af dine ~16 features)
  sample_size = 0.7   
) |> 
  parsnip::set_engine("xgboost") |> 
  parsnip::set_mode("classification")

regime_wf_light <- workflows::workflow() |> 
  workflows::add_recipe(regime_recipe) |> 
  workflows::add_model(xgb_spec_light)

# fitting af modellen ----

cl <- parallel::makePSOCKcluster(3) 
doParallel::registerDoParallel(cl)

regime_results_light <- tune::fit_resamples(
  object       = regime_wf_light, 
  resamples    = regime_folds_light,
  metrics      = yardstick::metric_set(roc_auc, accuracy),
  control      = tune::control_resamples(save_pred = TRUE, verbose = TRUE)
)

parallel::stopCluster(cl)

# Resultater ----
tune::collect_metrics(regime_results_light)
preds <- tune::collect_predictions(regime_results_light)


# vi fitter nu på hele train og predicter på test ----

final_model <- parsnip::fit(regime_wf_light, data = train_data)

final_model %>%
  extract_fit_parsnip() %>%
  vip::vip(geom = "point", num_features = 10) +
  labs(title = "Hvad styrer modellen?", subtitle = "Hvis YieldCurve eller RateShort ligger øverst, er det årsagen.")



test_results <- test_data %>%
  dplyr::select(date, sp500) %>%
  dplyr::bind_cols(
    predict(final_model, new_data = test_data)
  ) %>%
  dplyr::bind_cols(
    predict(final_model, new_data = test_data, type = "prob")
  )

head(test_results)

# regn afkast ----
strategy_fixed <- test_results %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    bh_ret = sp500 / dplyr::lag(sp500) - 1,
    
    # Hent sandsynligheden for BULL marked
    prob_bull = .pred_bull, 
    
    # NY LOGIK: Vær kun i cash, hvis Bull-sandsynligheden er meget lav (< 35%)
    # Dvs. vi kræver > 65% sandsynlighed for Bear for at sælge.
    trade_signal = ifelse(prob_bull > 0.35, "bull", "bear"),
    
    # Husk at lagge signalet (vi handler i morgen på dagens signal)
    final_signal = dplyr::lag(trade_signal),
    
    model_ret = ifelse(final_signal == "bull", bh_ret, 0)
  ) %>%
  dplyr::filter(!is.na(bh_ret), !is.na(model_ret)) %>%
  dplyr::mutate(
    cum_bh = cumprod(1 + bh_ret),
    cum_model = cumprod(1 + model_ret)
  )

# Plot den nye kurve
library(ggplot2)
strategy_fixed %>%
  tidyr::pivot_longer(cols = c(cum_bh, cum_model), names_to = "Strategy", values_to = "Value") %>%
  ggplot(aes(x = date, y = Value, color = Strategy)) +
  geom_line(linewidth = 1) +
  scale_color_manual(
    values = c("cum_bh" = "black", "cum_model" = "red"),
    labels = c("Buy & Hold", "XGBoost (Threshold 35%)")
  ) +
  theme_minimal() +
  labs(
    title = "XGBoost Strategi med Justeret Tærskel",
    subtitle = "Modellen går kun i cash, hvis Bear-sandsynligheden er > 65%",
    y = "Vækst af 1 krone",
    x = "Dato"
  )
