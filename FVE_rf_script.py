import numpy as np
import pandas as pd
import os
import sys
import time


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



def rf_bootstrap(exp_name, m=100, seed=None, n_estimators=100, partial=False, partial_tsa=False):
    if partial:
        nickname = f"{exp_name}_{m}_{seed}_partial"
    elif partial_tsa:
        nickname = f"{exp_name}_{m}_{seed}_partial_tsa"
    else:
        nickname = f"{exp_name}_{m}_{seed}"

    output_file= f"rf_output/{nickname}_output.txt"
    os.makedirs("rf_output", exist_ok=True)

    with open(output_file, "w") as f:
        f.write("Writing model output... \n")
        print("Writing model output... \n")
        f.flush()

        # load data
        print(f"reading in data at: {time.ctime(time.time())}", file=f)
        if m == 1:
            if partial == True:
                test_dat = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_test_data_partial.csv")
                train_dat = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_train_data_partial.csv")
                x_cols = [col for col in test_dat.columns if ("_r" in col) or ("_l" in col)]
            elif partial_tsa == True:
                test_dat = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_test_data_partial_tsa.csv")
                train_dat = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_train_data_partial_tsa.csv")
                x_cols = [col for col in test_dat.columns if ("_r" in col) or ("_l" in col)]
            else:
                test_dat = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_test_data.csv")
                train_dat = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/reg_train_data.csv")
                x_cols = [col for col in test_dat.columns if ("_r" in col) or ("_l" in col)]
                x_cols.append("interview_age")
                x_cols.append("sex_2")
            
            X_boot = train_dat.dropna()[x_cols]
            X_oob = test_dat.dropna()[x_cols]
            y_boot = train_dat.dropna()['nihtbx_cryst_uncorrected']
            y_oob = test_dat.dropna()['nihtbx_cryst_uncorrected']

            X_boot = X_boot.reset_index(drop=True)
            y_boot = y_boot.reset_index(drop=True)
            X_oob = X_oob.reset_index(drop=True)
            y_oob = y_oob.reset_index(drop=True)
            print(f"sanity check: partial={partial}, partial_tsa={partial_tsa}, m={m}, dim_x = {X_boot.shape}, {X_oob.shape}; dim_y = {y_boot.shape}, {y_oob.shape}; x_cols={x_cols[0:1]} to {x_cols[20483:len(x_cols)]}")



        else:
            if partial == True:
                FVE_df = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/FVE_dat_partial.csv")
                x_cols = [col for col in FVE_df.columns if ("_r" in col) or ("_l" in col)]
            elif partial_tsa == True:
                FVE_df = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/FVE_dat_partial_tsa.csv")
                x_cols = [col for col in FVE_df.columns if ("_r" in col) or ("_l" in col)]
            else:
                FVE_df = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/FVE_dat.csv")
                x_cols = [col for col in FVE_df.columns if ("_r" in col) or ("_l" in col)]
                x_cols.append("interview_age")
                x_cols.append("sex_2")
            
            FVE_df_nona = FVE_df.dropna()

            X = FVE_df_nona[x_cols]
            y = FVE_df_nona['nihtbx_cryst_uncorrected']
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            print(f"sanity check: partial={partial}, partial_tsa={partial_tsa}, m={m}, len(x_cols)={len(x_cols)}, x_cols={x_cols[0:1]} to {x_cols[20483:len(x_cols)]}", file=f, flush=True)
        

        print(f"data done at: {time.ctime(time.time())}", file=f, flush=True)
        print(f"data done at: {time.ctime(time.time())}")
        f.flush()

        mse_list, r2_list = [], []

        print(f"Starting {m} bootstrap experiments, seed={seed}, n_estimators={n_estimators}", file=f, flush=True)
        print(f"Starting {m} bootstrap experiments, seed={seed}, n_estimators={n_estimators}")
        f.flush()
        rng = np.random.RandomState(seed)

        for i in range(m):
            print(f"Bootstrap sample {i+1}/{m} at: {time.ctime(time.time())}", file=f, flush=True)
            print(f"Bootstrap sample {i+1}/{m} at: {time.ctime(time.time())}")
            f.flush()

            # separate out X_boot and X_oob
            if m != 1:
                n = len(X)
                iter_rng = np.random.RandomState(seed + i) if seed is not None else np.random
                boot_indices = iter_rng.choice(n, size=n, replace=True)
                oob_indices = np.setdiff1d(np.arange(n), boot_indices)

                if len(oob_indices) == 0:
                    continue

                X_boot, y_boot = X.iloc[boot_indices], y.iloc[boot_indices]
                X_oob, y_oob = X.iloc[oob_indices], y.iloc[oob_indices]

            rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=None if seed is None else seed + i)
            rf_regressor.fit(X_boot, y_boot)

            y_pred = rf_regressor.predict(X_oob)
            mse = mean_squared_error(y_oob, y_pred)
            r2 = r2_score(y_oob, y_pred)

            mse_list.append(mse)
            r2_list.append(r2)

            if m != 1:
                print(f"Bootstrap {i+1}/{m}: OOB size={len(oob_indices)}, MSE={mse:.4f}, R2={r2:.4f}, finished at {time.ctime(time.time())}", file=f)
                print(f"Bootstrap {i+1}/{m}-Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}", file=f)
                print(f"Bootstrap {i+1}/{m}-Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}", file=f)
                f.flush()
            else:
                print(f"Standard Split-Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}", file=f)
                print(f"Standard Split-Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}", file=f)
                f.flush()
            
            #pd.Series(mse_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{nickname}_MSE.csv")
            pd.Series(r2_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{nickname}_R2.csv")

        print("\nFinal Bootstrap Results", file=f)
        print(f"Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}", file=f)
        print(f"Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}", file=f)
        
        print(f"Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}")
        print(f"Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}")
        
        #pd.Series(mse_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{nickname}_MSE.csv")
        pd.Series(r2_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{nickname}_R2.csv")
        print(f"Finished all runs ({time.ctime(time.time())})", file=f)
        print(f"Finished all runs ({time.ctime(time.time())})")
        f.flush()



def main(exp_name, m=100, seed=None, n_estimators=100, partial=False, partial_tsa=False):
    rf_bootstrap(exp_name=exp_name, m=m, seed=seed, n_estimators=n_estimators, partial=partial, partial_tsa=partial_tsa)


if __name__ == "__main__":
    exp_name = sys.argv[1]
    partial = (sys.argv[2] == "True")
    partial_tsa = (sys.argv[3] == "True")
    n_estimators = int(sys.argv[4])
    seed = int(sys.argv[5]) if sys.argv[4] != "None" else None
    m = int(sys.argv[6])

    main(exp_name=exp_name, partial=partial, seed=seed, n_estimators=n_estimators, m=m, partial_tsa=partial_tsa)
