import numpy as np
import pandas as pd
import os
import sys
import time


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



def rf_bootstrap(exp_name, m=100, seed=None, n_estimators=100, partial=False):
    output_file= f"rf_output/{exp_name}_output.txt"
    os.makedirs("rf_output", exist_ok=True)

    with open(output_file, "w") as f:
        f.write("Writing model output... \n")
        print("Writing model output... \n")
        f.flush()

        # load data
        print(f"reading in data at: {time.ctime(time.time())}", file=f)
        if partial == True:
            FVE_df = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/FVE_df.csv")
        else:
            FVE_df = pd.read_csv("/niddk-data-central/mae_hr/FVE/data_out/FVE_dat_partial.csv")
        FVE_df_nona = FVE_df.dropna()
        

        x_cols = [col for col in FVE_df_nona.columns if ("_r" in col) or ("_l" in col)]
        x_cols = [col for col in x_cols if ("total" not in col)]
        X = FVE_df_nona[x_cols + ['interview_age', 'demo_sex_v2.x']]
        y = FVE_df_nona['nihtbx_cryst_uncorrected']
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        print(f"data done at: {time.ctime(time.time())}", file=f)
        print(f"data done at: {time.ctime(time.time())}")
        f.flush()

        mse_list, r2_list = [], []

        print(f"Starting {m} bootstrap experiments", file=f)
        print(f"Starting {m} bootstrap experiments")
        f.flush()
        n = len(X)
        rng = np.random.RandomState(seed)

        for i in range(m):
            print(f"Bootstrap sample {i+1}/{m} at: {time.ctime(time.time())}", file=f)
            print(f"Bootstrap sample {i+1}/{m} at: {time.ctime(time.time())}")
            f.flush()
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

            print(f"Bootstrap {i+1}/{m}: OOB size={len(oob_indices)}, MSE={mse:.4f}, R2={r2:.4f}, finished at {time.ctime(time.time())}", file=f)
            print(f"Bootstrap {i+1}/{m}-Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}", file=f)
            print(f"Bootstrap {i+1}/{m}-Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}", file=f)
            f.flush()
            pd.Series(mse_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{exp_name}_MSE.csv")
            pd.Series(r2_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{exp_name}_R2.csv")

        print("\nFinal Bootstrap Results", file=f)
        print(f"Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}", file=f)
        print(f"Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}", file=f)
        
        print(f"Mean MSE: {np.mean(mse_list):.4f} +/- {np.std(mse_list):.4f}")
        print(f"Mean R2 : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}")
        
        pd.Series(mse_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{exp_name}_MSE.csv")
        pd.Series(r2_list).to_csv(f"/niddk-data-central/mae_hr/FVE/rf_output/{exp_name}_R2.csv")
        print(f"Finished all runs ({time.ctime(time.time())})", file=f)
        print(f"Finished all runs ({time.ctime(time.time())})")
        f.flush()



def main(exp_name, m=100, seed=None, n_estimators=100, partial=False):
    rf_bootstrap(exp_name=exp_name, m=m, seed=seed, n_estimators=n_estimators, partial=partial)


if __name__ == "__main__":
    exp_name = sys.argv[1]
    partial = (sys.argv[2] == "True")
    n_estimators = int(sys.argv[3])
    seed = int(sys.argv[4]) if sys.argv[3] != "None" else None
    m = int(sys.argv[5])

    main(exp_name=exp_name, partial=partial, seed=seed, n_estimators=n_estimators, m=m)
