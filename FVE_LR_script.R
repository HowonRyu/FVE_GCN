#install.packages(c("tidyverse", "GGally", "broom", "glmnet", "rsample"),lib="/data/howon/FVE/Rlibs", repos = "https://cloud.r-project.org/")
#install.packages(c("glmnet"),lib="/data/howon/FVE/Rlibs", repos = "https://cloud.r-project.org/")
#.libPaths("/usr/lib/R/library")
#install.packages(c("broom", "GGally", "glmnet", "rsample", "readr"))
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

wd="/niddk-data-central/mae_hr/FVE"
#wd="~/Projects/FVE"


######################### LASSO and ridge #########################
# For bootstrapping
reg_data_org_all = readr::read_csv(file.path(wd, "data_out", "FVE_dat.csv")) 
reg_data_org_partial = readr::read_csv(file.path(wd, "data_out", "FVE_dat_partial.csv")) 
reg_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "FVE_dat_partial_tsa.csv"))

dim(reg_data_org_all)
dim(reg_data_org_partial)


# For standard split
reg_train_data_org = readr::read_csv(file.path(wd, "data_out", "reg_train_data.csv"))
reg_test_data_org  = readr::read_csv(file.path(wd, "data_out", "reg_test_data.csv"))
reg_train_data_org_partial = readr::read_csv(file.path(wd, "data_out", "reg_train_data_partial.csv"))
reg_test_data_org_partial = readr::read_csv(file.path(wd, "data_out", "reg_test_data_partial.csv"))
reg_train_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "reg_train_data_partial_tsa.csv"))
reg_test_data_org_partial_tsa = readr::read_csv(file.path(wd, "data_out", "reg_test_data_partial_tsa.csv"))


dim(reg_train_data_org)
dim(reg_test_data_org)
dim(reg_train_data_org_partial)
dim(reg_test_data_org_partial)

##########################  Var estimation Bootstrap  ########################## 

# Bootstrap sampling
B = 1
set.seed(1004)

if (B !=1) {
  boot_samples <- bootstraps(reg_data_org_all, times = B)
  boot_samples_p <- bootstraps(reg_data_org_partial, times = B)
  boot_samples_p_tsa <- bootstraps(reg_data_org_partial_tsa, times = B)

  test_ratio = 0.2
  n = nrow(reg_data_org_all)
  n1 = as.integer(n*(1-test_ratio))
  n2 = n-n1
}

# initialize loop
LASSO_test_r2 = c()
Ridge_test_r2 = c()
LASSO_partial_test_r2 = c()
Ridge_partial_test_r2 = c()
LASSO_partial_tsa_test_r2 = c()
Ridge_partial_tsa_test_r2 = c()


all_models <- list()


# Loop through each bootstrap resample
for (b in 1:B) {
  print(paste0("B=",b,"/", B," start at ",Sys.time()))


  #regular
  if (B==1) {
    print("B=1, using standard test/train split")
    reg_train_data = reg_train_data_org
    reg_test_data = reg_test_data_org
  } else {
    split <- boot_samples$splits[[b]]
    reg_train_data <- analysis(split)      # In-bag
    oob_samples  <- assessment(split)
    reg_test_data <- oob_samples %>% slice_sample(n = n2, replace = TRUE)
  }
  
  X_var_mat_train =  as.matrix(reg_train_data %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_train =  as.matrix(reg_train_data %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_train = as.matrix(X_var_mat_train[complete.cases(X_var_mat_train),])
  y_var_mat_train = as.matrix(y_var_mat_train[complete.cases(X_var_mat_train),])
  
  X_var_mat_test =  as.matrix(reg_test_data %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_test =  as.matrix(reg_test_data %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_test = as.matrix(X_var_mat_test[complete.cases(X_var_mat_test),])
  y_var_mat_test = as.matrix(y_var_mat_test[complete.cases(X_var_mat_test),])


  # partial
  if (B==1) {
    print("B=1, using standard test/train split")
    reg_train_data_partial = reg_train_data_org_partial
    reg_test_data_partial = reg_test_data_org_partial
  } else {
    split_p <- boot_samples_p$splits[[b]]
    reg_train_data_partial <- analysis(split_p)      # In-bag
    oob_samples_p  <- assessment(split_p)
    reg_test_data_partial <- oob_samples_p %>% slice_sample(n = n2, replace = TRUE)
  }
  
  X_var_mat_train_p =  as.matrix(reg_train_data_partial %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_train_p =  as.matrix(reg_train_data_partial %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_train_p = as.matrix(X_var_mat_train_p[complete.cases(X_var_mat_train_p),])
  y_var_mat_train_p = as.matrix(y_var_mat_train_p[complete.cases(X_var_mat_train_p),])
  
  X_var_mat_test_p =  as.matrix(reg_test_data_partial %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_test_p =  as.matrix(reg_test_data_partial %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_test_p = as.matrix(X_var_mat_test_p[complete.cases(X_var_mat_test_p),])
  y_var_mat_test_p = as.matrix(y_var_mat_test_p[complete.cases(X_var_mat_test_p),])


  # partial_tsa
  if (B==1) {
    print("B=1, using standard test/train split")
    reg_train_data_partial_tsa = reg_train_data_org_partial_tsa
    reg_test_data_partial_tsa = reg_test_data_org_partial_tsa
  } else {
    split_p <- boot_samples_p_tsa$splits[[b]]
    reg_train_data_partial_tsa <- analysis(split_p)      # In-bag
    oob_samples_p  <- assessment(split_p)
    reg_test_data_partial_tsa <- oob_samples_p %>% slice_sample(n = n2, replace = TRUE)
  }
  
  X_var_mat_train_p_tsa =  as.matrix(reg_train_data_partial_tsa %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_train_p_tsa =  as.matrix(reg_train_data_partial_tsa %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_train_p_tsa = as.matrix(X_var_mat_train_p_tsa[complete.cases(X_var_mat_train_p_tsa),])
  y_var_mat_train_p_tsa = as.matrix(y_var_mat_train_p_tsa[complete.cases(X_var_mat_train_p_tsa),])

  X_var_mat_test_p_tsa =  as.matrix(reg_test_data_partial_tsa %>% select(-nihtbx_cryst_uncorrected))
  y_var_mat_test_p_tsa =  as.matrix(reg_test_data_partial_tsa %>% select(nihtbx_cryst_uncorrected))
  X_var_mat_test_p_tsa = as.matrix(X_var_mat_test_p_tsa[complete.cases(X_var_mat_test_p_tsa),])
  y_var_mat_test_p_tsa = as.matrix(y_var_mat_test_p_tsa[complete.cases(X_var_mat_test_p_tsa),])




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

  # LASSO partial_tsa
  LASSO_cv_fit_p_tsa <- cv.glmnet(X_var_mat_train_p_tsa, y_var_mat_train_p_tsa, alpha = 1, nfolds = 5)
  LASSO_opt_lambda_p_tsa <- LASSO_cv_fit_p_tsa$lambda.min
  LASSO_model_p_tsa <- glmnet(X_var_mat_train_p_tsa, y_var_mat_train_p_tsa, alpha = 1, lambda = LASSO_opt_lambda_p_tsa)  # LASSO with best lambd
  LASSO_pred_test_p_tsa = predict(LASSO_model_p_tsa, s = LASSO_opt_lambda_p_tsa, newx = X_var_mat_test_p_tsa)
  LASSO_partial_tsa_test_r2 = c(LASSO_partial_tsa_test_r2, my_R2(true=y_var_mat_test_p_tsa, pred=LASSO_pred_test_p_tsa))      
  all_models[[paste0("LASSO_partial_tsa_b", b)]] <- LASSO_model_p_tsa
  
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

  # Ridge partial_tsa
  Ridge_cv_fit_p_tsa <- cv.glmnet(X_var_mat_train_p_tsa, y_var_mat_train_p_tsa, alpha = 0, nfolds = 5)
  Ridge_opt_lambda_p_tsa <- Ridge_cv_fit_p_tsa$lambda.min
  Ridge_model_p_tsa <- glmnet(X_var_mat_train_p_tsa, y_var_mat_train_p_tsa, alpha = 0, lambda = Ridge_opt_lambda_p_tsa)  # Ridge with best lambda
  Ridge_pred_test_p_tsa = predict(Ridge_model_p_tsa, s = Ridge_opt_lambda_p_tsa, newx = X_var_mat_test_p_tsa)
  Ridge_partial_tsa_test_r2 =c(Ridge_partial_tsa_test_r2, my_R2(true=y_var_mat_test_p_tsa, pred=Ridge_pred_test_p_tsa))
  all_models[[paste0("Ridge_partial_tsa_b", b)]] <- Ridge_model_p_tsa

}


#save the models
if (B==1) {
  save(all_models, file=paste0(wd, "/LR_output/LR_all_models_standard.Rdata"))
} else {
  save(all_models, file=paste0(wd, "/LR_output/LR_all_models_boot", B, ".Rdata"))
}




# organize outcomes
result_dict_boot <- list(
  LASSO_test_r2 = LASSO_test_r2, LASSO_partial_test_r2 = LASSO_partial_test_r2, LASSO_partial_tsa_test_r2 = LASSO_partial_tsa_test_r2,
  Ridge_test_r2 = Ridge_test_r2, Ridge_partial_test_r2 = Ridge_partial_test_r2, Ridge_partial_tsa_test_r2 = Ridge_partial_tsa_test_r2)

#result_dict_boot <- list(LASSO_partial_tsa_test_r2 = LASSO_partial_tsa_test_r2, Ridge_partial_tsa_test_r2 = Ridge_partial_tsa_test_r2)

result_tbl_boot <- data.frame(
  variable = names(result_dict_boot),
  mean = sapply(result_dict_boot, mean),
  var_boot   = sapply(result_dict_boot, var),   
  sd_boot =   sapply(result_dict_boot, sd)
)
result_tbl_boot[, c(2,4)] <- round(result_tbl_boot[, c(2,4)], 4)


# save the result
if (B==1) {
  save(result_tbl_boot, file=paste0(wd, "/LR_output/LR_result_tbl_standard.Rdata"))
} else {
  save(result_tbl_boot, file=paste0(wd, "/LR_output/LR_result_tbl_boot", B, ".Rdata"))
}




