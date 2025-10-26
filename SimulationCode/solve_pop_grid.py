"""
This script solves for the population distribution under different
model parameterizations, looping over a grid of parameter values.
"""

# imports
# %%
import numpy as np
import pandas as pd
import importlib.resources
import os
import json
from ogcore.parameters import Specifications
from ogcore.utils import safe_read_pickle, shift_bio_clock
from ogcore import demographics as demog


# %%
# Find baseline population distribution
cur_dir = os.path.dirname(os.path.realpath(__file__))
sim_dir = os.path.join(cur_dir, "simulation_results")
base_dir = os.path.join(sim_dir, "baseline")
"""
Read in OG-USA default parameters
"""
with importlib.resources.open_text(
    "ogusa", "ogusa_default_parameters.json"
) as file:
    ogusa_default_params = json.load(file)

# update some of these defaults that will be used in all simulations
ogusa_default_params["tax_func_type"] = "HSV"
ogusa_default_params["tG1"] = 30
ogusa_default_params["RC_TPI"] = 1e-04

"""
---------------------------------------------------------------------------
Baseline parameters
---------------------------------------------------------------------------
"""
# Set up baseline parameterization
num_workers = 1
p = Specifications(
    baseline=True,
    num_workers=num_workers,
    baseline_dir=base_dir,
    output_base=base_dir,
)
p.update_specifications(ogusa_default_params)

# Update demographics using OG-Core demographics module
# And find baseline demographic objects not returned by get_pop_objs
if os.path.exists(os.path.join(sim_dir, "demog_vars_baseline.pkl")):
    (
        fert_rates_baseline_TP,
        mort_rates_baseline_TP,
        infmort_rates_baseline_TP,
        imm_rates_baseline_TP,
        pop_baseline_TP,
        pre_pop_dist_baseline,
    ) = safe_read_pickle(os.path.join(sim_dir, "demog_vars_baseline.pkl"))
else:
    fert_rates_baseline = demog.get_fert(
        start_year=p.start_year, end_year=p.start_year
    )
    mort_rates_baseline, infmort_rates_baseline = demog.get_mort(
        start_year=p.start_year, end_year=p.start_year
    )
    pop_dist_baseline, pre_pop_baseline = demog.get_pop(
        start_year=p.start_year, end_year=p.start_year
    )
    imm_rates_baseline = demog.get_imm_rates(
        fert_rates=fert_rates_baseline,
        mort_rates=mort_rates_baseline,
        infmort_rates=infmort_rates_baseline,
        pop_dist=pop_dist_baseline,
        start_year=p.start_year,
        end_year=p.start_year,
    )
    # And extend each fof these over the full time path
    fert_rates_baseline_TP = np.append(
        fert_rates_baseline,
        np.tile(
            fert_rates_baseline[-1, :].reshape((1, p.E + p.S)),
            (p.T + p.S - 2, 1),
        ),
        axis=0,
    )
    mort_rates_baseline_TP = np.append(
        mort_rates_baseline,
        np.tile(
            mort_rates_baseline[0, :].reshape((1, p.E + p.S)),
            (p.T + p.S - 2, 1),
        ),
        axis=0,
    )
    infmort_rates_baseline_TP = np.append(
        infmort_rates_baseline[0],
        np.ones(p.T + p.S - 1) * infmort_rates_baseline[0],
    )
    imm_rates_baseline_TP = np.append(
        imm_rates_baseline,
        np.tile(
            imm_rates_baseline[0, :].reshape((1, p.E + p.S)),
            (p.T + p.S - 2, 1),
        ),
        axis=0,
    )
    pop_baseline_TP, pre_pop_dist_baseline = demog.get_pop(
        infer_pop=True,
        fert_rates=fert_rates_baseline_TP,
        mort_rates=mort_rates_baseline_TP,
        infmort_rates=infmort_rates_baseline_TP,
        imm_rates=imm_rates_baseline_TP,
        initial_pop=None,
        pre_pop_dist=None,
        start_year=p.start_year,
        end_year=p.start_year + 1,
    )
# Now get population objects for the model
num_periods = 2100 - p.start_year  # 2100 is the last year WPP forecast
demog_vars = demog.get_pop_objs(
    p.E,
    p.S,
    p.T,
    fert_rates=fert_rates_baseline_TP[:num_periods, :],
    mort_rates=mort_rates_baseline_TP[:num_periods, :],
    infmort_rates=infmort_rates_baseline_TP[:num_periods],
    imm_rates=imm_rates_baseline_TP[:num_periods, :],
    infer_pop=True,
    pop_dist=pop_baseline_TP[:2, :],
    pre_pop_dist=pre_pop_dist_baseline,
    initial_data_year=p.start_year,
    final_data_year=p.start_year + num_periods - 1,
)
p.update_specifications(demog_vars, raise_errors=False)
# No find the full baseline population distribution until 2126
end_year = 2126
base_pop_full_path, _ = demog.get_pop(
    E=20,
    S=80,
    min_age=0,
    max_age=99,
    infer_pop=True,
    fert_rates=fert_rates_baseline_TP,
    mort_rates=mort_rates_baseline_TP,
    infmort_rates=infmort_rates_baseline_TP,
    imm_rates=imm_rates_baseline_TP,
    initial_pop=pop_baseline_TP[0, :],
    pre_pop_dist=pre_pop_dist_baseline,
    start_year=p.start_year,
    end_year=end_year,
    download_path=None,
)

# Create dictionary to store results
results_dict = {
    "age_effect": [],
    "initial_effect": [],
    "final_effect": [],
    "mort_effect": [],
    "fert_effect": [],
    "pop_diffs_2045_2065": [],
    "pop_diffs_2026_2100": [],
    "pop_diffs_2050": [],
    "total_pop_diff_2050": [],
}

# Create grid of parameters to loop over
parameter_grid = {
    "age_effect": [40, 50, 60, 70, 80],
    "initial_effect": [0, 5, 10, 15],
    "final_effect": [0, 5, 10, 15],
    "mort_effect": [0, 0.2, 0.5, 1, 2.5, 5, 7, 9, 10],
    "fert_effect": [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.6],
}
total_number_of_sims = (
    len(parameter_grid["age_effect"])
    * len(parameter_grid["initial_effect"])
    * len(parameter_grid["final_effect"])
    * len(parameter_grid["mort_effect"])
    * len(parameter_grid["fert_effect"])
)
print(f"Total number of simulations to run: {total_number_of_sims}")

# Now loop over parameter grid and find population distribution at each point
i = 1
for min_age_effect_felt in parameter_grid["age_effect"]:
    for initial_effect_period in parameter_grid["initial_effect"]:
        for final_effect in parameter_grid["final_effect"]:
            for mort_effect in parameter_grid["mort_effect"]:
                for fert_effect in parameter_grid["fert_effect"]:

                    # if both mort_effect and fert_effect are zero, skip
                    if mort_effect == 0 and fert_effect == 0:
                        continue

                    print(f"Running simulation {i} of {total_number_of_sims}")
                    print(
                        f"Parameters: min_age_effect_felt={min_age_effect_felt}, "
                        f"initial_effect_period={initial_effect_period}, "
                        f"final_effect_period={final_effect}, "
                        f"mort_effect={mort_effect}, "
                        f"fert_effect={fert_effect}"
                    )
                    i += 1
                    # Store parameter values
                    results_dict["age_effect"].append(min_age_effect_felt)
                    results_dict["initial_effect"].append(
                        initial_effect_period
                    )
                    results_dict["final_effect"].append(final_effect)
                    results_dict["mort_effect"].append(mort_effect)
                    results_dict["fert_effect"].append(fert_effect)
                    # Create parameter object
                    p = Specifications(
                        baseline=False,
                        num_workers=num_workers,
                        baseline_dir=base_dir,
                    )
                    p.update_specifications(ogusa_default_params)
                    # update to baseline demographics
                    # important for shift of rho below
                    p.update_specifications(demog_vars)

                    # create final effect period that is the sum of
                    # initial_effect and final_effect
                    final_effect_period = initial_effect_period + final_effect

                    # Updates to mortality rates and fertility rates
                    mort_rates_shift = shift_bio_clock(
                        p.rho.copy(),
                        initial_effect_period=initial_effect_period,
                        final_effect_period=final_effect_period,
                        total_effect=mort_effect,
                        min_age_effect_felt=min_age_effect_felt - p.E,
                        bound_above=True,
                    )
                    mort_rates_adjusted = mort_rates_baseline_TP.copy()
                    mort_rates_adjusted[:, p.E :] = mort_rates_shift[:-1, :]
                    mort_rates_adjusted[:, -1] = (
                        1  # make sure last period is 1
                    )

                    fert_rates_shift = shift_bio_clock(
                        fert_rates_baseline_TP.copy(),
                        initial_effect_period=initial_effect_period,
                        final_effect_period=final_effect_period,
                        total_effect=fert_effect,
                        min_age_effect_felt=min_age_effect_felt,
                        bound_below=True,
                    )

                    # Find population distribution under new parameters
                    sim_pop_path, _ = demog.get_pop(
                        E=20,
                        S=80,
                        min_age=0,
                        max_age=99,
                        infer_pop=True,
                        fert_rates=fert_rates_shift,
                        mort_rates=mort_rates_adjusted,
                        infmort_rates=infmort_rates_baseline_TP,
                        imm_rates=imm_rates_baseline_TP,
                        initial_pop=pop_baseline_TP[0, :],
                        pre_pop_dist=pre_pop_dist_baseline,
                        start_year=p.start_year,
                        end_year=2126,
                        download_path=None,
                    )

                    # Compute differences in the  population distributions
                    pop_diff = sim_pop_path[24, :] - base_pop_full_path[24, :]
                    results_dict["total_pop_diff_2050"].append(
                        pop_diff.sum() / 1_000_000
                    )
                    results_dict["pop_diffs_2045_2065"].append(
                        (
                            sim_pop_path[20:40, 20:]
                            - base_pop_full_path[20:40, 20:]
                        ).sum()
                        / 1_000_000
                    )
                    results_dict["pop_diffs_2026_2100"].append(
                        (
                            sim_pop_path[:76, 20:]
                            - base_pop_full_path[:76, 20:]
                        ).sum()
                        / 1_000_000
                    )
                    results_dict["pop_diffs_2050"].append(
                        (
                            sim_pop_path[24, 20:] - base_pop_full_path[24, 20:]
                        ).sum()
                        / 1_000_000
                    )
# Put results dictionary in a dataframe
results_df = pd.DataFrame(results_dict)
# Save dataframe to csv
results_df.to_csv(
    os.path.join(sim_dir, "population_grid_results.csv"), index=False
)
