# %%
# imports
import pandas as pd
import numpy as np
import argparse
import os
import json
from ogcore.utils import safe_read_pickle, unstationarize_vars
from ogcore.demographics import get_pop

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# constants
QALY = 160_000  # Quality adjusted life year value, in dollars
NUM_YEARS_MACRO_NPV = 50  # TODO: update to 100 # Number of years to compute NPV of macro variables
NUM_YEARS_NPV = 100  # TODO remove the above when pop is for 100 years...
AVG_START_YEAR = 2046  # start year for 20 year averages
AVG_END_YEAR = 2066  # end year for 20 year averages
END_YEAR_NPV = 2126  # Last year to compute NPV of macro variables


def main(directory_name, json_filename, OUTPUT_FILENAME):
    # %%
    # Set some variables and constants
    directory = os.path.join(CUR_DIR, directory_name)
    # %%
    # Get baseline TPI results, parameters, and demographics
    base_path = os.path.join(directory, "baseline")
    base_vars = safe_read_pickle(
        os.path.join(base_path, "TPI", "TPI_vars.pkl")
    )
    # base_params = safe_read_pickle(os.path.join(base_path, "model_params.pkl"))
    base_params = safe_read_pickle(os.path.join(base_path, "model_params.pkl"))
    (
        fert_rates_baseline_TP,
        mort_rates_baseline_TP,
        infmort_rates_baseline_TP,
        imm_rates_baseline_TP,
        pop_dist_base_TP,
        pre_pop_dist_baseline,
    ) = safe_read_pickle(os.path.join(directory, "demog_vars_baseline.pkl"))
    tot_pop_2026 = pop_dist_base_TP[
        0, 20:
    ].sum()  # TODO: should we just do working age?  growth rates, g_n just refer to these...
    base_pop_path = np.cumprod(1 + base_params.g_n) * (
        tot_pop_2026 / 1_000_000
    )
    # Compute expected lifetime income for cohort born in 2026
    # take main diagonal of y_before_tax_mat
    income_profiles = np.zeros((base_params.S, base_params.J))
    for j in range(base_params.J):
        income_profiles[:, j] = base_vars["before_tax_income"][
            20:, :, j
        ].diagonal()
    # compute mean income across all ability types
    base_mean_income_profile = (
        income_profiles * base_params.lambdas.reshape(1, 10)
    ).sum(axis=1)

    # %%
    # Read in CBO long-term forecast
    cbo_lt_forecast = pd.read_csv(
        os.path.join(
            CUR_DIR,
            "..",
            "Data",
            "cbo_long_term_forecast_extended.csv",
        )
    )
    # Drop GDP
    cbo_lt_forecast.drop(columns=["GDP"], inplace=True)
    # Rename RGDP to GDP
    cbo_lt_forecast.rename(columns={"RGDP": "GDP"}, inplace=True)
    # Put RGDP in 2026 dollars
    cbo_lt_forecast["GDP"] = cbo_lt_forecast["GDP"] * (28.9 / 21)

    # %%
    results_dict = {
        "Sim Name": [],
        "age_effect": [],
        "initial_effect": [],
        "final_effect": [],
        "mort_effect": [],
        "prod_effect": [],
        "fert_effect": [],
        "pop_diffs_2045_2065": [],
        "pop_diffs_2026_2100": [],
        "pop_diffs_2050": [],
        "total_pop_diff_2050": [],
        "avg_diff": [],
        "avg_gdp_pc_diff": [],
        "discount_rate": [],
        "NPV": [],
        "QALY 20, $B": [],
        "QALY NPV, $T": [],
    }
    no_npv_list = list(results_dict.keys())
    # Remove any element with NPV in the name
    no_npv_list = [k for k in no_npv_list if "NPV" not in k]
    no_npv_list.remove("discount_rate")  # remove discount rate
    # Read in JSON with metadata on simulations
    sim_info = json.load(open(os.path.join(CUR_DIR, json_filename)))
    for k, v in sim_info["scenario_params"].items():
        # Put the numbers in the results dictionary
        results_dict["age_effect"].append(v["min_age_effect_felt"])
        results_dict["initial_effect"].append(v["initial_effect_period"])
        results_dict["final_effect"].append(v["final_effect_period"])
        results_dict["mort_effect"].append(v["mort_effect"])
        results_dict["prod_effect"].append(v["prod_effect"])
        results_dict["fert_effect"].append(v["fert_effect"])

        results_dict["Sim Name"].append(k)

        file_path = os.path.join(directory, v["directory"])
        # Read in TPI results
        tpi_vars = safe_read_pickle(
            os.path.join(file_path, "TPI", "TPI_vars.pkl")
        )
        params = safe_read_pickle(os.path.join(file_path, "model_params.pkl"))

        # Compute differences in the  population
        end_year = 2100 - 1
        base_pop_full_path, _ = get_pop(
            E=20,
            S=80,
            min_age=0,
            max_age=99,
            infer_pop=True,
            fert_rates=fert_rates_baseline_TP,
            mort_rates=mort_rates_baseline_TP,
            infmort_rates=infmort_rates_baseline_TP,
            imm_rates=imm_rates_baseline_TP,
            initial_pop=pop_dist_base_TP[0, :],
            pre_pop_dist=pre_pop_dist_baseline,
            start_year=base_params.start_year,
            end_year=end_year,
            download_path=None,
        )
        sim_pop_path = np.loadtxt(
            os.path.join(
                file_path, "demographic_data", "population_distribution.csv"
            ),
            delimiter=",",
        )
        pop_diff = sim_pop_path[24, :] - base_pop_full_path[24, :]
        results_dict["total_pop_diff_2050"].append(pop_diff.sum() / 1_000_000)
        results_dict["pop_diffs_2045_2065"].append(
            (sim_pop_path[20:40, 20:] - base_pop_full_path[20:40, 20:]).sum()
            / 1_000_000
        )
        results_dict["pop_diffs_2026_2100"].append(
            (sim_pop_path[:76, 20:] - base_pop_full_path[:76, 20:]).sum()
            / 1_000_000
        )
        results_dict["pop_diffs_2050"].append(
            (sim_pop_path[24, 20:] - base_pop_full_path[24, 20:]).sum()
            / 1_000_000
        )

        # Compute percentage changes in GDP in model
        reformY = unstationarize_vars(
            "Y",
            tpi_vars,
            params,
        )
        baseY = unstationarize_vars(
            "Y",
            base_vars,
            base_params,
        )
        ans = (reformY - baseY) / baseY
        # Put results in DataFrame for merge
        pct_changes = pd.DataFrame(ans, columns=["Y"])
        pct_changes.reset_index(names="Year", inplace=True)
        pct_changes["Year"] = pct_changes["Year"] + base_params.start_year
        # Keep just years start year to end year
        pct_changes = pct_changes.loc[
            (pct_changes["Year"] >= base_params.start_year)
            & (pct_changes["Year"] <= END_YEAR_NPV)
        ]
        # compute percentage changes in macro variables
        # merge with CBO long-term forecast
        df = pct_changes.merge(cbo_lt_forecast, how="inner", on="Year")
        # Create new GDP series for reform scenario
        df["GDP_new"] = df["GDP"] * (1 + df["Y"])
        # Create difference in GDP series
        df["GDP_diff"] = df["GDP_new"] - df["GDP"]
        # Create GDP per capita series
        num_periods = 2100 - base_params.start_year
        df["GDP_pc"] = (df["GDP"][:num_periods] * 1e12) / base_pop_full_path[
            :num_periods, 20:
        ].sum(axis=1)
        df["GDP_pc_new"] = (df["GDP_new"][:num_periods] * 1e12) / sim_pop_path[
            :num_periods, 20:
        ].sum(axis=1)
        df["GDP_pc_diff"] = df["GDP_pc_new"] - df["GDP_pc"]
        results_dict["avg_gdp_pc_diff"].append(
            np.mean(
                df.loc[
                    (df["Year"] >= AVG_START_YEAR)
                    & (df["Year"] <= AVG_END_YEAR),
                    "GDP_pc_diff",
                ]
            )
        )
        # Compute average difference in GDP over 20 years
        results_dict["avg_diff"].append(
            np.mean(
                df.loc[
                    (df["Year"] >= AVG_START_YEAR)
                    & (
                        df["Year"] <= AVG_END_YEAR - 1
                    ),  # TODO: remove -1 or not?
                    "GDP_diff",
                ]
            )
            * 1000
        )  # to put in billions

        # Compute QALY effects
        life_years_added = (sim_pop_path - base_pop_full_path).sum(axis=1)
        value_of_life_years = life_years_added * QALY
        start_year = AVG_START_YEAR - params.start_year
        end_year = AVG_END_YEAR - params.start_year
        results_dict["QALY 20, $B"].append(
            value_of_life_years[start_year : end_year + 1].mean()
            / 1_000_000_000
        )  # put in billions

        # Compute NPV of difference in GDP series
        # for r in np.linspace(0.01, 0.06, 6):
        for i, r in enumerate([0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06]):
            npv = (
                df["GDP_diff"]
                / ((1 + r) ** (df["Year"] - base_params.start_year))
            ).sum()
            # Add values to dictionary
            results_dict["discount_rate"].append(r)
            results_dict["NPV"].append(npv)

            # create 1+r series
            r_series = np.array([1 + r] * len(value_of_life_years))
            # Create a NPV of changes over NUM_YEARS_MACRO_NPV years
            PV_value_of_life_years = (
                value_of_life_years / np.cumprod(r_series)
            ).sum()
            # Put in trillions of dollars
            PV_value_of_life_years /= 1e12
            results_dict["QALY NPV, $T"].append(PV_value_of_life_years)
            if i > 0:
                # extend all items in dictionary by 1 beyond the first r
                # Need to not extend discount_rate or NPV
                for key in no_npv_list:
                    results_dict[key].append(results_dict[key][-1])

    # Create DataFrame from dictionary
    results_df = pd.DataFrame(results_dict)
    # Save DataFrame to disk
    results_df.to_csv(os.path.join(CUR_DIR, OUTPUT_FILENAME), index=False)
    # %%


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory_name",
        help="Name of directory containing simulation results",
    )  # positional arg
    parser.add_argument(
        "json_filename",
        help="Name of JSON file with simulation metadata",
    )  # positional arg
    parser.add_argument(
        "output_filename", help="Name for CSV file with processed results"
    )  # positional arg
    args = parser.parse_args()
    main(args.directory_name, args.json_filename, args.output_filename)
