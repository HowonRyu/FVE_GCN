#install.packages(c("tidyverse", "GGally", "broom", "glmnet", "rsample"),lib="/data/howon/FVE/Rlibs", repos = "https://cloud.r-project.org/")
#install.packages(c("MASS"),lib="/data/howon/FVE/Rlibs", repos = "https://cloud.r-project.org/")
.libPaths("/data/howon/FVE/Rlibs")
library(dplyr)
library(GGally)
library(broom)
library(glmnet)
library(rsample)
library(MASS)


my_R2 = function(true, pred) {
  ss_res <- sum((true - pred)^2)
  ss_tot <- sum((true - mean(true))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  return(r_squared)
}

######################### LASSO and ridge #########################
###### Data load
reg_train_data_org = readr::read_csv("data_out/reg_train_all_data.csv") 
reg_test_data_org = readr::read_csv("data_out/reg_test_all_data.csv") 

c3_exclude = TRUE
if (c3_exclude) {
  reg_train_data = reg_train_data_org[reg_train_data_org$demo_sex_v2.x != 3,] %>%
    dplyr::select(-`...1`, -total_surface_area, -demo_sex_v2.x, -sex_3) #exclude sex=3 9,080 rows
  reg_test_data = reg_test_data_org[reg_test_data_org$demo_sex_v2.x != 3,]  %>%
    dplyr::select(-`...1`, -total_surface_area, -demo_sex_v2.x, -sex_3) #exclude sex=3 2271 rows, no sex3
  
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
    dplyr::select(-`...1`, -total_surface_area, -demo_sex_v2.x) 
  reg_test_data = reg_test_data_org %>%
    dplyr::select(-`...1`, -total_surface_area, -demo_sex_v2.x) 
  
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
reg_data_org_all_before = rbind(reg_train_data, reg_test_data)
reg_data_org_partial_before = rbind(reg_train_data_partial, reg_test_data_partial)


downsample_skip = 2
downsampled_cols = c(paste0(seq(1,10242, downsample_skip),"_l"), paste0(seq(1,10242, downsample_skip),"_r"))

#downsample_rows = dim(reg_data_org_all_before)[1]
downsample_rows = 3000
reg_data_org_all = reg_data_org_all_before[1:downsample_rows,] %>% dplyr::select(c(downsampled_cols, "nihtbx_cryst_uncorrected"))
reg_data_org_partial = reg_data_org_partial_before[1:downsample_rows,]  %>% dplyr::select(c(downsampled_cols, "nihtbx_cryst_uncorrected"))


dim(reg_data_org_all)
dim(reg_data_org_partial)




##########################  Var estimation Bootstrap  ########################## 
print(paste0("Starting at ",Sys.time()))
# Bootstrap sampling
B = 50
set.seed(1004)
boot_samples <- bootstraps(reg_data_org_all, times = B)
boot_samples_p <- bootstraps(reg_data_org_partial, times = B)

test_ratio = 0.2
n = nrow(reg_data_org_all)
n1 = as.integer(n*(1-test_ratio))
n2 = n-n1
    

# initialize loop
pinv_R2_test = c()
pinv_R2_test_p = c()

# Loop through each bootstrap resample
for (b in 1:B) {
  print(paste0("B=",b,"/", B," start at ",Sys.time()))
  
  #regular
  split <- boot_samples$splits[[b]]
  reg_train_data <- analysis(split)      # In-bag
  oob_samples  <- assessment(split)
  reg_test_data <- oob_samples %>% slice_sample(n = n2, replace = TRUE)

  print(paste0("Data prep started running at:", Sys.time()))
  X_var_mat_train =  as.matrix(reg_train_data %>% dplyr::select(-nihtbx_cryst_uncorrected))
  X_var_mat_train = cbind(Intercept = 1, X_var_mat_train)
  y_var_mat_train =  as.matrix(reg_train_data %>% dplyr::select(nihtbx_cryst_uncorrected))
  X_var_mat_train = as.matrix(X_var_mat_train[complete.cases(X_var_mat_train),])
  y_var_mat_train = as.matrix(y_var_mat_train[complete.cases(X_var_mat_train),])
  
  X_var_mat_test =  as.matrix(reg_test_data %>% dplyr::select(-nihtbx_cryst_uncorrected))
  X_var_mat_test = cbind(Intercept = 1, X_var_mat_test)
  y_var_mat_test =  as.matrix(reg_test_data %>% dplyr::select(nihtbx_cryst_uncorrected))
  X_var_mat_test = as.matrix(X_var_mat_test[complete.cases(X_var_mat_test),])
  y_var_mat_test = as.matrix(y_var_mat_test[complete.cases(X_var_mat_test),])
   
  
  # partial
  split_p <- boot_samples_p$splits[[b]]
  reg_train_data_partial <- analysis(split_p)      # In-bag
  oob_samples_p  <- assessment(split_p)
  reg_test_data_partial <- oob_samples_p %>% slice_sample(n = n2, replace = TRUE)
  
  X_var_mat_train_p =  as.matrix(reg_train_data_partial %>% dplyr::select(-nihtbx_cryst_uncorrected))
  X_var_mat_train_p = cbind(Intercept = 1, X_var_mat_train_p)
  y_var_mat_train_p =  as.matrix(reg_train_data_partial %>% dplyr::select(nihtbx_cryst_uncorrected))
  X_var_mat_train_p = as.matrix(X_var_mat_train_p[complete.cases(X_var_mat_train_p),])
  y_var_mat_train_p = as.matrix(y_var_mat_train_p[complete.cases(X_var_mat_train_p),])
  
  X_var_mat_test_p =  as.matrix(reg_test_data_partial %>% dplyr::select(-nihtbx_cryst_uncorrected))
  X_var_mat_test_p = cbind(Intercept = 1, X_var_mat_test_p)
  y_var_mat_test_p =  as.matrix(reg_test_data_partial %>% dplyr::select(nihtbx_cryst_uncorrected))
  X_var_mat_test_p = as.matrix(X_var_mat_test_p[complete.cases(X_var_mat_test_p),])
  y_var_mat_test_p = as.matrix(y_var_mat_test_p[complete.cases(X_var_mat_test_p),])
  
  
  print(paste0("Regular started running at:", Sys.time()))
  #PINV
  X = X_var_mat_train
  y = y_var_mat_train
  X_test = X_var_mat_test
  y_test = y_var_mat_test

  pi_beta = ginv(X) %*% y
    
  y_pred_train <- X %*% pi_beta
  y_pred_test <- X_test %*% pi_beta
    
  pinv_R2_test = c(pinv_R2_test, my_R2(true=y_test, pred=y_pred_test))

  print(paste0("Partial started running at:", Sys.time()))
  #PINV partial
  X_p = X_var_mat_train_p
  y_p = y_var_mat_train_p
  X_test_p = X_var_mat_test_p
  y_test_p = y_var_mat_test_p

  pi_beta_p = ginv(X_p) %*% y_p
    
  y_pred_train_p <- X_p %*% pi_beta_p
  y_pred_test_p <- X_test_p %*% pi_beta_p
    

  pinv_R2_test_p = c(pinv_R2_test_p, my_R2(true=y_test_p, pred=y_pred_test_p))

  
}


# organize outcomes
result_dict_boot <- list(
  pinv_R2_test = pinv_R2_test, pinv_R2_test_p = pinv_R2_test_p)

result_tbl_boot <- data.frame(
  variable = names(result_dict_boot),
  mean = sapply(result_dict_boot, mean),
  var_boot   = sapply(result_dict_boot, var),   
  sd_boot =   sapply(result_dict_boot, sd)
)
result_tbl_boot[, c(2,4)] <- round(result_tbl_boot[, c(2,4)], 4)


# Print the result
save(result_tbl_boot, file="/data/howon/FVE/run_output/PINV_result_tbl_boot_ds.Rdata")
print(paste0("Finished at ",Sys.time()))




