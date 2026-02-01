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
  recipes::update_role(
    DateTime, 
    new_role = "id"
  ) |>
  
  # 2. FEATURE ENGINEERING: DOMAIN KNOWLEDGE
  recipes::step_mutate(
    ResidualLoad = SystemLoadEA - ForecastWindProduction,
    WindRatio = ForecastWindProduction / SystemLoadEA
  ) |>
  
  # 3. Target Transformation
  recipes::step_YeoJohnson(SMPEP2) |>
  
  # 4. Tids-features
  recipes::step_date(DateTime, features = c("dow", "month", "year"), label = FALSE) |>
  recipes::step_time(DateTime, features = c("hour")) |>
  recipes::step_harmonic(DateTime_hour, frequency = 24, cycle_size = 24) |>
  recipes::step_harmonic(DateTime_month, frequency = 12, cycle_size = 12) |>
  recipes::step_harmonic(DateTime_dow, frequency = 7, cycle_size = 7) |>
  
  # 5. LAGS
  recipes::step_lag(SMPEP2, lag = c(24, 48, 168)) |>
  recipes::step_lag(
    ActualWindProduction, 
    SystemLoadEP2, 
    ORKTemperature, 
    ORKWindspeed, 
    lag = 24
  ) |>
  
  # 6. ROLLING FEATURES
  # Vi bruger step_window fra embed pakken på den laggede værdi
  embed::step_window(
    SMPEP2_lag_024, 
    size = 24, 
    statistic = c("mean", "sd"),
    role = "predictor",
    names = paste0("SMPEP2_24h_rolling_", c("mean", "sd"))
  ) |>
  
  # 7. CLEANUP
  recipes::step_rm(
    ActualWindProduction, 
    SystemLoadEP2, 
    ORKTemperature, 
    ORKWindspeed
  ) |>
  
  # 8. Standard rensning
  recipes::step_naomit(recipes::all_predictors()) |>
  recipes::step_novel(recipes::all_nominal_predictors()) |> 
  recipes::step_dummy(recipes::all_nominal_predictors()) |>
  recipes::step_zv(recipes::all_predictors())
  
# specificere model (tuning) ----
xgb_spec <- parsnip::boost_tree(
  trees = 10, 
  tree_depth = tune::tune(), 
  min_n = tune::tune(), 
  loss_reduction = 0.001, 
  sample_size = 0.7,
  mtry = tune::tune(), 
  learn_rate = tune::tune()
) |> 
  parsnip::set_engine("xgboost") |> 
  parsnip::set_mode("regression")

# definer workflow ----
xgb_wf <- workflows::workflow() |> 
  workflows::add_recipe(energy_recipe) |> 
  workflows::add_model(xgb_spec)

# definer grid ----
# Vi skal definere søgerummet, ellers ved tune_race ikke hvor den skal lede
xgb_grid <- dials::grid_latin_hypercube(
  dials::tree_depth(),
  dials::min_n(),
  dials::learn_rate(),
  # mtry skal tilpasses antallet af predictors (her ca 20-30)
  dials::mtry(range = c(5, 25)), 
  size = 10 # Antal kandidater
)

# fit på resamples (Racing) ----
cores <- parallel::detectCores() - 1
cl <- parallel::makePSOCKcluster(cores)
doParallel::registerDoParallel(cl)

progressr::with_progress(
  expr = {
    xgb_race_results <- finetune::tune_race_anova(
      xgb_wf,
      resamples = folds,
      grid = xgb_grid, # Husk at give grid med!
      metrics = yardstick::metric_set(
        yardstick::rmse,
        yardstick::rsq
      ),
      control = finetune::control_race(
        verbose_elim = TRUE, 
        verbose = T,
        save_pred = TRUE, 
        parallel_over = "everything"
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
