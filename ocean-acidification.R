library(tidyverse)
library(tidymodels)
library(xgboost)

set.seed(123)

dat_train <- read_csv(here::here('data', 'ocean-chemistry', 'train.csv')) %>% 
  janitor::clean_names() %>%
  select(-c(x13)) %>% 
  rename(ta1 = ta1_x)

dat_test <- read_csv(here::here('data', 'ocean-chemistry', 'test.csv')) %>% 
  janitor::clean_names()

# Split data into training set (70%) and test set (30%)
ocean_split <- rsample::initial_split(dat_train, prop = 0.70, strata = 'dic')

# Data preprocessing ----
ocean_recipe <- recipes::recipe(dic ~ ., data = dat_train) %>%
  # Recode
  recipes::step_normalize() %>%
  # Prep recipe
  recipes::prep(dat_train)

# Define 10-fold cross-validation and stratify based on outcome variable
cv_fold <- dat_train %>% 
  rsample::vfold_cv(v = 10, strata = 'dic')

# Model building ----
## Hyperparameter tuning: learning rate ----
# Define model specification for learning rate
learn_rate_spec <- parsnip::boost_tree(learn_rate = tune()) %>%
  # Set tuning software package
  parsnip::set_engine('xgboost') %>%
  # Set type of prediction outcome
  parsnip::set_mode('regression')

# Define parameters grid for learning rate
learn_rate_grid <- expand.grid(learn_rate = seq(0.0001, 0.3, length.out = 30))

# Define workflow for learning rate
learn_rate_wf <- workflows::workflow() %>%
  # Add model specification
  workflows::add_model(learn_rate_spec) %>%
  # Add recipe
  workflows::add_recipe(ocean_recipe)

# Tune model
tictoc::tic() 
learn_rate_tune <- learn_rate_wf %>%
  tune::tune_grid(resamples = cv_folds, grid = learn_rate_grid)
tictoc::toc() 

# Select best model based on root-mean-square deviation metric
learn_rate_best_rmse <- tune::select_best(learn_rate_tune, metric = 'rmse')

# Extract learning rate associated with best model
learn_rate_best <- learn_rate_best_rmse$learn_rate

## Hyperparameter tuning: tree parameters ----
# Define model specification for tree parameters
tree_params_spec <- parsnip::boost_tree(learn_rate = learn_rate_best,
                                        trees = tune(),
                                        tree_depth = tune(),
                                        min_n = tune(),
                                        loss_reduction = tune()) %>%
  # Set tuning software package
  parsnip::set_engine('xgboost') %>%
  # Set type of prediction outcome
  parsnip::set_mode('regression')

# Define parameters
tree_params <- dials::parameters(trees(), tree_depth(), min_n(), loss_reduction())

# Define parameters grid for tree parameters
tree_params_grid <- dials::grid_max_entropy(tree_params, size = 20, iter = 100)

# Define workflow for tree parameters
tree_params_wf <- workflows::workflow() %>%
  # Add model specification
  workflows::add_model(tree_params_spec) %>%
  # Add recipe
  workflows::add_recipe(ocean_recipe)

# Tune model
tictoc::tic() 
tree_params_tune <- tree_params_wf %>%
  tune::tune_grid(resamples = cv_folds, grid = tree_params_grid)
tictoc::toc() 

# Check performance
tree_params_tune %>% collect_metrics() 

# Select best model based on root-mean-square deviation metric
tree_params_best_rmse <- select_best(tree_params_tune, metric = 'rmse')

# Extract number of trees associated with best model
trees_best <- tree_params_best_rmse$trees

# Extract maximum tree depth associated with best model
tree_depth_best <- tree_params_best_rmse$tree_depth

# Extract minimum n associated with best model
min_n_best <- tree_params_best_rmse$min_n

# Extract reduction in loss function (gamma) associated with best model
loss_reduction_best <- tree_params_best_rmse$loss_reduction

## Hyperparameter tuning: stochastic parameters ----
# Define model specification for stochastic parameters
stochastic_params_spec <- parsnip::boost_tree(learn_rate = learn_rate_best,
                                              trees = trees_best,
                                              tree_depth = tree_depth_best,
                                              min_n = min_n_best,
                                              loss_reduction = loss_reduction_best,
                                              mtry = tune(),
                                              sample_size = tune()) %>% 
  # Set tuning software package
  parsnip::set_engine('xgboost') %>%
  # Set type of prediction outcome
  parsnip::set_mode('regression')

# Define parameters
stochastic_params <- dials::parameters(finalize(mtry(), select(ocean_chem_train, -dic)),
                                       sample_size = sample_prop(c(0.4, 0.9)))

# Define parameters grid for stochastic parameters
stochastic_params_grid <- dials::grid_max_entropy(stochastic_params, size = 20, iter = 100)

# Define workflow for stochastic parameters
stochastic_params_wf <- workflows::workflow() %>%
  # Add model specification
  workflows::add_model(stochastic_params_spec) %>%
  # Add recipe
  workflows::add_recipe(ocean_recipe)

# Tune model
tictoc::tic() 
stochastic_params_tune <- stochastic_params_wf %>%
  tune::tune_grid(resamples = cv_folds, grid = stochastic_params_grid)
tictoc::toc() 

# Check performance
stochastic_params_tune %>% collect_metrics() 

# Select best model based on root-mean-square deviation metric
stochastic_params_best_rmse <- tune::select_best(stochastic_params_tune, metric = 'rmse')

# Extract number of randomly sampled predictors associated with best model
mtry_best <- stochastic_params_best_rmse$mtry

# Extract sample size associated with best model
sample_size_best <- stochastic_params_best_rmse$sample_size

# Model selection ----
final_spec <-  parsnip::boost_tree(learn_rate = learn_rate_best,
                                   trees = trees_best,
                                   tree_depth = tree_depth_best,
                                   min_n = min_n_best,
                                   loss_reduction = loss_reduction_best,
                                   mtry = mtry_best,
                                   sample_size = sample_size_best) %>%
  # Set tuning software package
  parsnip::set_engine('xgboost') %>%
  # Set type of prediction outcome
  parsnip::set_mode('regression')

# Define final workflow
final_wf <- workflows::workflow() %>%
  # Add model specification
  workflows::add_model(final_spec) %>%
  # Add recipe
  workflows::add_recipe(ocean_recipe)

# Fit workflow to initial split
final_fit <- tune::last_fit(final_wf, ocean_split)

# Check performance
final_fit %>% tune::collect_metrics()

# View predictions
final_fit$.predictions

# Extract predictions
final_predictions <- final_fit %>% tune::collect_predictions()

# Model evaluation ----
# Fit workflow to training set
final_training <- final_wf %>% 
  parsnip::fit(data = dat_train)

# Make predictions on test set
final_test <- final_training %>% 
  stats::predict(new_data = dat_test) 

dat_predict_test <- dat_test %>%
  cbind(predictions$.pred) %>%
  rename(dic = `predictions$.pred`)

# View variable importance
final_wf %>% 
  parsnip::fit(data = dat_predict_test) %>%
  tune::extract_fit_parsnip() %>% vip::vip()

# Dataset submission ----
dat_predict <- dat_test %>%
  cbind(predictions$.pred) %>%
  rename(DIC = `predictions$.pred`, `test.id` = id) %>%
  select(c(`test.id`, DIC))

write_csv(dat_predict, here::here('data', 'ocean-chemistry', 'predict.csv'))