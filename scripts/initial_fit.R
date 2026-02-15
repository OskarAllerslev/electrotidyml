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
energy_recipe <- recipes::recipe(SMPEP2 ~ ., data = train_df) |>
  # 1. ID og Roller
  
  # 2. FEATURE ENGINEERING
  recipes::step_mutate(
    ResidualLoad = SystemLoadEA - ForecastWindProduction,
    WindRatio = ForecastWindProduction / SystemLoadEA
  ) |>
  
  # 3. Tids-features
  recipes::step_date(DateTime, features = c("dow", "month", "year"), label = FALSE) |>
  recipes::step_time(DateTime, features = c("hour")) |>
  
  # Harmonics (Perioder baseret på integers fra date/time functions er OK)
  recipes::step_harmonic(DateTime_hour, frequency = 24, cycle_size = 24) |>
  recipes::step_harmonic(DateTime_month, frequency = 12, cycle_size = 12) |>
  recipes::step_harmonic(DateTime_dow, frequency = 7, cycle_size = 7) |>
  
  # 4. LAGS (Her retter vi til 30-minutters intervaller)
  # Vi vil have 24t, 48t og 1 uge tilbage.
  # 24 timer = 48 steps. 
  # 48 timer = 96 steps.
  # 1 uge    = 336 steps.
  recipes::step_lag(SMPEP2, lag = c(48, 96, 336)) |>
  
  # Ex-post variabler SKAL lagges med mindst 24 timer (48 steps) for at undgå leak
  recipes::step_lag(
    ActualWindProduction, 
    SystemLoadEP2, 
    ORKTemperature, 
    ORKWindspeed, 
    lag = 48
  ) |>
  
  # 5. ROLLING FEATURES
  # Vi skal bruge den laggede værdi fra 24 timer siden (SMPEP2_lag_048).
  # Size skal være ulige. 24 timer er 48 steps. Vi vælger 49 (ca. 24.5 timer)
  recipes::step_window(
    # Denne selector fanger "SMPEP2_lag_048" eller "SMPEP2_lag_48" sikkert
    tidyselect::matches("lag_48_SMPEP2"), 
    size = 49,  
    statistic = "mean",
    names = "SMPEP2_24h_rolling_mean"
  ) |>
  recipes::step_window(
    tidyselect::matches("lag_48_SMPEP2"), 
    size = 49,
    statistic = "sd",
    names = "SMPEP2_24h_rolling_sd"
  ) |>
  
  # 6. CLEANUP (Fjern alt ex-post data der ikke er lagget)
  recipes::step_rm(
    ActualWindProduction, 
    SystemLoadEP2, 
    ORKTemperature, 
    ORKWindspeed
  ) |>
  
  # 7. Standard rensning
  recipes::step_naomit(recipes::all_predictors()) |>
  recipes::step_novel(recipes::all_nominal_predictors()) |> 
  recipes::step_dummy(recipes::all_nominal_predictors()) |>
  recipes::step_zv(recipes::all_predictors())

# Kontrol: Prepper recipe for at sikre, at kolonnerne dannes korrekt
prep_check <- recipes::prep(energy_recipe, training = train_df)
print("Recipe er nu konfigureret korrekt til 30-minutters data.")
  
# specificere model (tuning) ----
#xgb_spec <- parsnip::boost_tree(
#  trees = 10, 
#  tree_depth = tune::tune(), 
#  min_n = tune::tune(), 
#  loss_reduction = 0.001, 
#  sample_size = 0.7,
#  mtry = tune::tune(), 
#  learn_rate = tune::tune()
#) |> 
#  parsnip::set_engine("xgboost") |> 
#  parsnip::set_mode("regression")

xgb_spec <- modeltime::arima_boost(
  mode = "regression", 
  seasonal_period = 48, 
  non_seasonal_ar = 0, 
  non_seasonal_differences = 1, 
  non_seasonal_ma = 1, 
  min_n = tune(), 
  tree_depth = tune(), 
  learn_rate = 0.1
) |> 
  parsnip::set_engine("arima_xgboost")



# definer workflow ----
xgb_wf <- workflows::workflow() |> 
  workflows::add_recipe(energy_recipe) |> 
  workflows::add_model(xgb_spec)

# definer grid ----
# Vi skal definere søgerummet, ellers ved tune_race ikke hvor den skal lede
library(modeltime)
xgb_grid <- dials::grid_regular(
  dials::tree_depth(range = c(3,8)), 
  dials::min_n(range = c(10,30)), 
  levels = 4
)

  

# fit på resamples (Racing) ----
cores <- parallel::detectCores(logical = FALSE) # Brug fysiske kerner
cl <- parallel::makePSOCKcluster(cores)

# HER ER FIXET: Vi tvinger hver kerne til at loade pakkerne
parallel::clusterEvalQ(cl, {
  library(tidymodels)
  library(modeltime)
  library(xgboost)
})

doParallel::registerDoParallel(cl)

# Så kører vi tuning
progressr::with_progress(
  expr = {
    xgb_race_results <- finetune::tune_race_anova(
      xgb_wf,
      resamples = folds,
      grid = xgb_grid,
      metrics = yardstick::metric_set(
        yardstick::rmse,
        yardstick::rsq
      ),
      control = finetune::control_race(
        verbose_elim = TRUE, 
        verbose = TRUE,
        save_pred = TRUE, 
        parallel_over = "everything" # Nu virker det, fordi pakkerne er loaded
      )
    )
  }
)

parallel::stopCluster(cl)

# resultater ----
# 1. Se resultaterne
tune::show_best(xgb_race_results, metric = "rmse")

# 2. Udvælg den bedste automatisk
best_params <- tune::select_best(xgb_race_results, metric = "rmse")
print(best_params)

fs::dir_create("output")
readr::write_rds(best_params, "output/best_params_xgb.rds")

# 3. Lås workflowet fast
final_wf <- xgb_wf |> 
  tune::finalize_workflow(best_params)

# 4. Last Fit på holdout data
final_fit <- tune::last_fit(final_wf, splits)

# A. Hent Lambda-værdien sikkert
trained_recipe <- final_fit |> 
  workflows::extract_recipe()
