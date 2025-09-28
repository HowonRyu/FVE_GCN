#install.packages(c("tidyverse", "GGally", "broom", "glmnet", "rsample"),lib="/data/howon/FVE/Rlibs", repos = "https://cloud.r-project.org/")
#install.packages(c("glmnet"),lib="/data/howon/FVE/Rlibs", repos = "https://cloud.r-project.org/")
.libPaths("/usr/lib/R/library")
install.packages(c("GGally", "glmnet", "rsample"))
library(dplyr)
library(GGally)
library(broom)
library(glmnet)
library(rsample)

my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

######################### LASSO and ridge #########################
###### Data load
reg_train_data_org = readr::read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_train_all_data.csv") 
reg_test_data_org = readr::read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_test_all_data.csv") 

c3_exclude = TRUE
if (c3_exclude) {
  reg_train_data = reg_train_data_org[reg_train_data_org$demo_sex_v2.x != 3,] %>%
    select(-`...1`, -total_surface_area, -demo_sex_v2.x, -sex_3) #exclude sex=3 9,080 rows
  reg_test_data = reg_test_data_org[reg_test_data_org$demo_sex_v2.x != 3,]  %>%
    select(-`...1`, -total_surface_area, -demo_sex_v2.x, -sex_3) #exclude sex=3 2271 rows, no sex3
  
  # partial regression
  reg_train_data_partial = reg_train_data
  reg_test_data_partial = reg_test_data
  
  reg_train_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ interview_age + sex_2, data=reg_train_data)$residuals
  reg_test_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ interview_age + sex_2, data=reg_test_data)$residuals
  
  x_surface_cols = c(grep("_l", colnames(reg_train_data), value=TRUE), grep("_r", colnames(reg_train_data), value=TRUE))
  for (col in x_surface_cols) {
    reg_train_data_partial[[col]] = lm(reg_train_data[[col]] ~ reg_train_data$interview_age + reg_train_data$sex_2)$residuals
    reg_test_data_partial[[col]] = lm(reg_test_data[[col]] ~ reg_test_data$interview_age + reg_test_data$sex_2)$residuals
  }
  
} else {
  reg_train_data = reg_train_data_org %>%
    select(-`...1`, -total_surface_area, -demo_sex_v2.x) 
  reg_test_data = reg_test_data_org %>%
    select(-`...1`, -total_surface_area, -demo_sex_v2.x) 
  
  # partial regression
  reg_train_data_partial = reg_train_data
  reg_test_data_partial = reg_test_data
  
  reg_train_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ interview_age + sex_2 + sex_3, data=reg_train_data)$residuals
  reg_test_data_partial$nihtbx_cryst_uncorrected = lm(nihtbx_cryst_uncorrected ~ interview_age + sex_2 + sex_3, data=reg_test_data)$residuals
  
  x_surface_cols = c(grep("_l", colnames(reg_train_data), value=TRUE), grep("_r", colnames(reg_train_data), value=TRUE))
  for (col in x_surface_cols) {
    reg_train_data_partial[[col]] = lm(reg_train_data[[col]] ~ reg_train_data$interview_age +
                                         reg_train_data$sex_2 + reg_train_data$sex_3)$residuals
    reg_test_data_partial[[col]] = lm(reg_test_data[[col]] ~ reg_test_data$interview_age +
                                        reg_test_data$sex_2 + reg_test_data$sex_3)$residuals
  }
}

# combine train and test
reg_data_org_all = rbind(reg_train_data, reg_test_data)
reg_data_org_partial = rbind(reg_train_data_partial, reg_test_data_partial)




##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 100
set.seed(1004)
boot_samples <- bootstraps(reg_data_org_all, times = B)
boot_samples_p <- bootstraps(reg_data_org_partial, times = B)

test_ratio = 0.2
n = nrow(reg_data_org_all)
n1 = as.integer(n*(1-test_ratio))
n2 = n-n1
    

# initialize loop
LASSO_test_r2 = c()
Ridge_test_r2 = c()
LASSO_partial_test_r2 = c()
Ridge_partial_test_r2 = c()

all_models <- list()


# Loop through each bootstrap resample
for (b in 1:B) {
  print(paste0("B=",b,"/", B," start at ",Sys.time()))
  
  #regular
  split <- boot_samples$splits[[b]]
  reg_train_data <- analysis(split)      # In-bag
  oob_samples  <- assessment(split)
  reg_test_data <- oob_samples %>% slice_sample(n = n2, replace = TRUE)

  X_var_mat_train =  as.matrix(reg_train_data %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_train =  as.matrix(reg_train_data %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_train = as.matrix(X_var_mat_train[complete.cases(X_var_mat_train),])
  y_var_mat_train = as.matrix(y_var_mat_train[complete.cases(X_var_mat_train),])
  
  X_var_mat_test =  as.matrix(reg_test_data %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_test =  as.matrix(reg_test_data %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_test = as.matrix(X_var_mat_test[complete.cases(X_var_mat_test),])
  y_var_mat_test = as.matrix(y_var_mat_test[complete.cases(X_var_mat_test),])
  
  
  
  # partial
  split_p <- boot_samples_p$splits[[b]]
  reg_train_data_partial <- analysis(split_p)      # In-bag
  oob_samples_p  <- assessment(split_p)
  reg_test_data_partial <- oob_samples_p %>% slice_sample(n = n2, replace = TRUE)
  
  X_var_mat_train_p =  as.matrix(reg_train_data_partial %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_train_p =  as.matrix(reg_train_data_partial %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_train_p = as.matrix(X_var_mat_train_p[complete.cases(X_var_mat_train_p),])
  y_var_mat_train_p = as.matrix(y_var_mat_train_p[complete.cases(X_var_mat_train_p),])
  
  X_var_mat_test_p =  as.matrix(reg_test_data_partial %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_test_p =  as.matrix(reg_test_data_partial %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_test_p = as.matrix(X_var_mat_test_p[complete.cases(X_var_mat_test_p),])
  y_var_mat_test_p = as.matrix(y_var_mat_test_p[complete.cases(X_var_mat_test_p),])
  
  
  
  
  # LASSO
  LASSO_cv_fit <- cv.glmnet(X_var_mat_train, y_var_mat_train, alpha = 1, nfolds = 5)
  LASSO_opt_lambda <- LASSO_cv_fit$lambda.min
  LASSO_model <- glmnet(X_var_mat_train, y_var_mat_train, alpha = 1, lambda = LASSO_opt_lambda)  # LASSO with best lambda
  LASSO_pred_test = predict(LASSO_model, s = LASSO_opt_lambda, newx = X_var_mat_test)
  LASSO_test_r2 = c(LASSO_test_r2, my_R2(true=y_var_mat_test, pred=LASSO_pred_test))
  all_models[[paste0("LASSO_b", b)]] <- LASSO_model

  # LASSO partial
  LASSO_cv_fit_p <- cv.glmnet(X_var_mat_train_p, y_var_mat_train_p, alpha = 1, nfolds = 5)
  LASSO_opt_lambda_p <- LASSO_cv_fit_p$lambda.min
  LASSO_model_p <- glmnet(X_var_mat_train_p, y_var_mat_train_p, alpha = 1, lambda = LASSO_opt_lambda_p)  # LASSO with best lambd
  LASSO_pred_test_p = predict(LASSO_model_p, s = LASSO_opt_lambda_p, newx = X_var_mat_test_p)
  LASSO_partial_test_r2 = c(LASSO_partial_test_r2, my_R2(true=y_var_mat_test_p, pred=LASSO_pred_test_p))      
  all_models[[paste0("LASSO_partial_b", b)]] <- LASSO_model_p


  # Ridge
  Ridge_cv_fit <- cv.glmnet(X_var_mat_train, y_var_mat_train, alpha = 0, nfolds = 5)
  Ridge_opt_lambda <- Ridge_cv_fit$lambda.min
  Ridge_model <- glmnet(X_var_mat_train, y_var_mat_train, alpha = 0, lambda = Ridge_opt_lambda)  # Ridge with best lambda
  Ridge_pred_test = predict(Ridge_model, s = Ridge_opt_lambda, newx = X_var_mat_test)
  Ridge_test_r2 =c(Ridge_test_r2, my_R2(true=y_var_mat_test, pred=Ridge_pred_test))
  all_models[[paste0("Ridge_b", b)]] <- Ridge_model
  
  
  # Ridge partial
  Ridge_cv_fit_p <- cv.glmnet(X_var_mat_train_p, y_var_mat_train_p, alpha = 0, nfolds = 5)
  Ridge_opt_lambda_p <- Ridge_cv_fit_p$lambda.min
  Ridge_model_p <- glmnet(X_var_mat_train_p, y_var_mat_train_p, alpha = 0, lambda = Ridge_opt_lambda_p)  # Ridge with best lambda
  Ridge_pred_test_p = predict(Ridge_model_p, s = Ridge_opt_lambda_p, newx = X_var_mat_test_p)
  Ridge_partial_test_r2 =c(Ridge_partial_test_r2, my_R2(true=y_var_mat_test_p, pred=Ridge_pred_test_p))
  all_models[[paste0("Ridge_partial_b", b)]] <- Ridge_model_p
  
}

#save the models
save(all_models, file="/niddk-data-central/mae_hr/FVE/LR_output/LR_all_models_boot100.Rdata")

# organize outcomes
result_dict_boot <- list(
  LASSO_test_r2 = LASSO_test_r2, LASSO_partial_test_r2 = LASSO_partial_test_r2,
  Ridge_test_r2 = Ridge_test_r2, Ridge_partial_test_r2 = Ridge_partial_test_r2)

result_tbl_boot <- data.frame(
  variable = names(result_dict_boot),
  mean = sapply(result_dict_boot, mean),
  var_boot   = sapply(result_dict_boot, var),   
  sd_boot =   sapply(result_dict_boot, sd)
)
result_tbl_boot[, c(2,4)] <- round(result_tbl_boot[, c(2,4)], 4)


# save the result
save(result_tbl_boot, file="/niddk-data-central/mae_hr/FVE/LR_output/LR_result_tbl_boot100.Rdata")





