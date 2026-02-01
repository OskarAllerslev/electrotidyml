
# henter data ----
data <- readr::read_csv("data-raw/electricity.gui")

data_clean <- electrotidyml::clean_data(data = data)

# vi laver splits ----
set.seed(1)
splits <- rsample::initial_time_split(
  data = data_clean, 
  prop = 0.88
)

train_df <- rsample::training(splits)
test_df <- rsample::testing(splits)


# rolling cv ----

folds <- rsample::rolling_origin(
  data = train_df, 
  initial = 5000, 
  assess = 24 * 7,
  skip = 24 * 7,
  cumulative = TRUE
)

# sætter vores recipe op -----
energy_recipe <- recipes::recipe( SMPEP2 ~ ., data = train_df) |>
  recipes::update_role(
    DateTime, 
    ActualWindProduction, 
    SystemLoadEP2, 
    new_role = "id"
  ) |> 
 #  recipes::step_log(SMPEP2, offset = 1, skip = TRUE) |> 
  recipes::step_YeoJohnson(SMPEP2) |> 
  recipes::step_date(
    DateTime, 
    features = c("dow", "month", "year"), 
    label = FALSE
  ) |> 
  recipes::step_time(DateTime, features = c("hour")) |> 
  recipes::step_harmonic(DateTime_hour, frequency = 24, cycle_size = 24) |> 
  recipes::step_harmonic(DateTime_month, frequency = 12, cycle_size = 12) |> 
  recipes::step_harmonic(DateTime_dow, frequency = 7, cycle_size = 7) |> 
  recipes::step_lag(SMPEP2, lag = c(24, 48, 168)) |> 
  recipes::step_naomit(recipes::all_predictors()) |> 
  recipes::step_dummy(recipes::all_nominal_predictors()) |> 
  recipes::step_zv(recipes::all_predictors())
  
# specificere model ----

xgb_spec <- parsnip::boost_tree(
  trees = 1000, 
  tree_depth = 8, 
  min_n = 10, 
  loss_reduction = 0.001, 
  sample_size = 0.7,
  mtry = 15, 
  learn_rate = 0.05
) |> 
  parsnip::set_engine("xgboost") |> 
  parsnip::set_mode("regression")

# definer workflow ----

xgb_wf <- workflows::workflow() |> 
  workflows::add_recipe(energy_recipe) |> 
  workflows::add_model(xgb_spec)


# fit på resamples ----
# vi træner modellen X folds og regner gennemsnitlig fejl 

cores <- parallel::detectCores() -1

cl <- parallel::makePSOCKcluster(cores)

doParallel::registerDoParallel(cl)

progressr::with_progress(
  expr = {
    xgb_resamples <- tune::fit_resamples(
      xgb_wf,
      resamples = folds,
      metrics = yardstick::metric_set(
        yardstick::rmse,
        yardstick::rsq
      ),
      control = tune::control_resamples(save_pred = T, verbose = T)
    )
  }
)

parallel::stopCluster(cl)





# resultater ----

# Se gennemsnitlig performance på tværs af alle tids-vinduer

metrics_summary <- tune::collect_metrics(xgb_resamples)
print(metrics_summary)

# Plot predictions vs actuals for hver fold (for at se stabilitet over tid)
predictions <- tune::collect_predictions(xgb_resamples)

library(ggplot2)

p <- predictions |> 
  ggplot(aes(x = .pred, y = SMPEP2)) +
  geom_point(alpha = 0.1) +
  geom_abline(col = "red", lty = "dashed") +
  labs(
    title = "XGBoost Performance: Rolling Origin CV",
    subtitle = "Predicted vs Actual på tværs af alle tids-folds",
    x = "Predicted Price",
    y = "Actual Price"
  ) +
  theme_minimal()

print(p)


# så all in alle
# dette er en dårlig model, da den ikke kan predicte de tunge haler
# der er heller ikke laggede ting med 

# vi tjekker lige hele prediction ----

final_res <- tune::last_fit(xgb_wf, splits)

test_predictions <- tune::collect_predictions(final_res)

test_data_dates <- rsample::testing(splits) |> 
  dplyr::select(DateTime) |> 
  dplyr::mutate(.row = dplyr::row_number())

plot_data <- test_predictions |> 
  dplyr::bind_cols(test_data_dates |> dplyr::select(DateTime)) |> 
  tidyr::pivot_longer(
    cols = c(SMPEP2, .pred), 
    names_to = "Type", 
    values_to = "Price"
  ) |> 
  dplyr::mutate(
    Type = dplyr::case_when(
      Type == "SMPEP2" ~ "Actual",
      Type == ".pred" ~ "Predicted"
    )
  )

p_ts <- plot_data |> 
  dplyr::arrange(DateTime) |> 
  dplyr::slice_head(n = 24 * 14 * 2) |> 
  ggplot(aes(x = DateTime, y = Price, color = Type)) +
  geom_line(linewidth = 0.8, alpha = 0.8) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(
    title = "Time Series Forecast: Actual vs XGBoost",
    subtitle = "Viser performance på de første 14 dage af test-sættet (Holdout)",
    y = "Elpris (EUR/MWh)",
    x = "Tid"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(p_ts)

p_full <- plot_data |> 
  ggplot(aes(x = DateTime, y = Price, color = Type)) +
  geom_line(alpha = 0.5, linewidth = 0.3) +
  scale_color_manual(values = c("Actual" = "black", "Predicted" = "red")) +
  labs(title = "Full Test Set Performance") +
  theme_minimal()

print(p_full)



