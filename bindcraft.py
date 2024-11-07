"""Main entry script for running BindCraft."""

import argparse
import os
import shutil
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pyrosetta as pr
from Bio import BiopythonWarning
from functions.biopython_utils import target_pdb_rmsd, validate_design_sequence
from functions.colabdesign_utils import (
    binder_hallucination,
    calc_ss_percentage,
    calculate_clash_score,
    clear_mem,
    copy_dict,
    masked_binder_predict,
    mk_afdesign_model,
    mpnn_gen_sequence,
    pr_relax,
    predict_binder_alone,
)
from functions.generic_utils import (
    TrajectoryData,
    calculate_averages,
    check_accepted_designs,
    check_filters,
    check_jax_gpu,
    check_n_trajectories,
    create_dataframe,
    generate_dataframe_labels,
    generate_directories,
    generate_filter_pass_csv,
    insert_data,
    load_af2_models,
    load_helicity,
    load_json_settings,
    perform_advanced_settings_check,
    perform_input_check,
    save_fasta,
)
from functions.pyrosetta_utils import score_interface, unaligned_rmsd

# suppress warnings
# os.environ["SLURM_STEP_NODELIST"] = os.environ["SLURM_NODELIST"]
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=BiopythonWarning)


def parse_input_paths():
    """Parse command line arguments for configs of BindCraft."""
    parser = argparse.ArgumentParser(
        description="Script to run BindCraft binder design."
    )

    parser.add_argument(
        "--settings",
        "-s",
        type=str,
        required=True,
        help="Path to the basic settings.json file. Required.",
    )
    parser.add_argument(
        "--filters",
        "-f",
        type=str,
        default="./settings_filters/default_filters.json",
        help="Path to the filters.json file used to filter design. If not provided, default will be used.",
    )
    parser.add_argument(
        "--advanced",
        "-a",
        type=str,
        default="./settings_advanced/default_4stage_multimer.json",
        help="Path to the advanced.json file with additional design settings. If not provided, default will be used.",
    )

    return parser.parse_args()


def mpnn_optimize_trajectory(
    traj_data: TrajectoryData,
    prediction_models: list[int],
    design_paths: dict[str, str],
    target_settings: dict[str, Any],
    advanced_settings: dict[str, Any],
    filter_settings: dict[str, Any],
    mpnn_csv: str,
    mpnn_csv_labels: list[str],
    failure_csv: str,
    final_csv: str,
    multimer_validation: bool,
    binder_chain: str = "B",
):
    """Optimize the trajectory using ProteinMPNN (soluble weights).

    This function performs the following steps:
    1. Generates MPNN sequences for the given trajectory.
    2. Filters out sequences based on amino acid composition and duplicates.
    3. Compiles prediction models for apo and holo structures.
    4. Iterates over designed sequences to predict and score them.
    5. Calculates various statistics and scores for each sequence.
    6. Validates sequences against filter thresholds.
    7. Saves accepted sequences and their statistics to CSV files.

    Args:
        traj_data: Trajectory data containing information from `init_design_trajectory`.
        prediction_models: List of model numbers to use for structure prediction.
        design_paths: Dictionary containing paths for saving designs and statistics.
        target_settings: Dictionary containing target-specific settings.
        advanced_settings: Dictionary containing advanced design settings.
        filter_settings: Dictionary containing filter settings.
        mpnn_csv: Path to the CSV file for MPNN data.
        mpnn_csv_labels: Column labels for the MPNN CSV file.
        failure_csv: Path to the CSV file for logging failure statistics.
        final_csv: Path to the final CSV file for accepted designs.
        multimer_validation: Boolean indicating whether to use multimer validation.
        binder_chain: Chain identifier for the binder (default is "B"). This is a placeholder in case multi-chain parsing in ColabDesign gets changed.

    Returns:
        Number of accepted MPNN designs.
    """
    # initialise MPNN counters
    mpnn_n = 1
    accepted_mpnn = 0
    mpnn_dict = {}
    design_start_time = time.time()

    ### MPNN redesign of starting binder
    trajectory_pdb = os.path.join(design_paths["Trajectory"], traj_data.Design + ".pdb")

    mpnn_trajectories = mpnn_gen_sequence(
        trajectory_pdb,
        binder_chain,
        traj_data.InterfaceResidues,
        advanced_settings,
    )

    existing_mpnn_sequences = set(
        pd.read_csv(mpnn_csv, usecols=["Sequence"])["Sequence"].values
    )
    # create set of MPNN sequences with allowed amino acid composition
    restricted_AAs = (
        set(aa.strip().upper() for aa in advanced_settings["omit_AAs"].split(","))
        if advanced_settings["force_reject_AA"]
        else set()
    )
    mpnn_sequences = sorted(
        {
            mpnn_trajectories["seq"][n][-traj_data.Length :]: {
                "seq": mpnn_trajectories["seq"][n][-traj_data.Length :],
                "score": mpnn_trajectories["score"][n],
                "seqid": mpnn_trajectories["seqid"][n],
            }
            for n in range(advanced_settings["num_seqs"])
            if (
                not restricted_AAs
                or not any(
                    aa in mpnn_trajectories["seq"][n][-traj_data.Length :].upper()
                    for aa in restricted_AAs
                )
            )
            and mpnn_trajectories["seq"][n][-traj_data.Length :]
            not in existing_mpnn_sequences
        }.values(),
        key=lambda x: x["score"],
    )
    del existing_mpnn_sequences

    # check whether any sequences are left after amino acid rejection and duplication check, and if yes proceed with prediction
    if not mpnn_sequences:
        print(
            "Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation"
        )
        print("")
        return 0

    # add optimisation for increasing recycles if trajectory is beta sheeted
    if (
        advanced_settings["optimise_beta"]
        and float(traj_data.Binder_BetaSheet_percent) > 15
    ):
        advanced_settings["num_recycles_validation"] = advanced_settings[
            "optimise_beta_recycles_valid"
        ]
    ### Compile prediction models once for faster prediction of MPNN sequences
    clear_mem()
    # compile complex prediction model
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
    )
    complex_prediction_model.prep_inputs(
        pdb_filename=target_settings["starting_pdb"],
        chain=target_settings["chains"],
        binder_len=traj_data.Length,
        rm_target_seq=advanced_settings["rm_template_seq_predict"],
        rm_target_sc=advanced_settings["rm_template_sc_predict"],
    )
    # compile binder monomer prediction model
    binder_prediction_model = mk_afdesign_model(
        protocol="hallucination",
        use_templates=False,
        initial_guess=False,
        use_initial_atom_pos=False,
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
    )
    binder_prediction_model.prep_inputs(length=traj_data.Length)

    # iterate over designed sequences
    for mpnn_sequence in mpnn_sequences:
        mpnn_time = time.time()

        # compile sequences dictionary with scores and remove duplicate sequences
        if mpnn_sequence["seq"] in [v["seq"] for v in mpnn_dict.values()]:
            print("Skipping duplicate sequence")
            continue

        # generate mpnn design name numbering
        mpnn_design_name = traj_data.Design + "_mpnn" + str(mpnn_n)
        mpnn_score = round(mpnn_sequence["score"], 2)
        mpnn_seqid = round(mpnn_sequence["seqid"], 2)

        # add design to dictionary
        mpnn_dict[mpnn_design_name] = {
            "seq": mpnn_sequence["seq"],
            "score": mpnn_score,
            "seqid": mpnn_seqid,
        }

        # save fasta sequence
        if advanced_settings["save_mpnn_fasta"] is True:
            save_fasta(mpnn_design_name, mpnn_sequence["seq"], design_paths)

        ### Predict mpnn redesigned binder complex using masked templates
        mpnn_complex_statistics, pass_af2_filters = masked_binder_predict(
            complex_prediction_model,
            mpnn_sequence["seq"],
            mpnn_design_name,
            target_settings["starting_pdb"],
            target_settings["chains"],
            traj_data.Length,
            trajectory_pdb,
            prediction_models,
            advanced_settings,
            filter_settings,
            design_paths,
            failure_csv,
        )

        # if AF2 filters are not passed then skip the scoring
        if not pass_af2_filters:
            print(
                f"Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring"
            )
            mpnn_n += 1
            continue

        # calculate statistics for each model individually
        for model_num in prediction_models:
            mpnn_design_pdb = os.path.join(
                design_paths["MPNN"],
                f"{mpnn_design_name}_model{model_num+1}.pdb",
            )
            mpnn_design_relaxed = os.path.join(
                design_paths["MPNN/Relaxed"],
                f"{mpnn_design_name}_model{model_num+1}.pdb",
            )

            if os.path.exists(mpnn_design_pdb):
                # Calculate clashes before and after relaxation
                num_clashes_mpnn = calculate_clash_score(mpnn_design_pdb)
                num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)

                # analyze interface scores for relaxed af2 trajectory
                (
                    mpnn_interface_scores,
                    mpnn_interface_AA,
                    mpnn_interface_residues,
                ) = score_interface(mpnn_design_relaxed, binder_chain)

                # secondary structure content of starting trajectory binder
                (
                    mpnn_alpha,
                    mpnn_beta,
                    mpnn_loops,
                    mpnn_alpha_interface,
                    mpnn_beta_interface,
                    mpnn_loops_interface,
                    mpnn_i_plddt,
                    mpnn_ss_plddt,
                ) = calc_ss_percentage(mpnn_design_pdb, advanced_settings, binder_chain)

                # unaligned RMSD calculate to determine if binder is in the designed binding site
                rmsd_site = unaligned_rmsd(
                    trajectory_pdb,
                    mpnn_design_pdb,
                    binder_chain,
                    binder_chain,
                )

                # calculate RMSD of target compared to input PDB
                target_rmsd = target_pdb_rmsd(
                    mpnn_design_pdb,
                    target_settings["starting_pdb"],
                    target_settings["chains"],
                )

                # add the additional statistics to the mpnn_complex_statistics dictionary
                mpnn_complex_statistics[model_num + 1].update(
                    {
                        "i_pLDDT": mpnn_i_plddt,
                        "ss_pLDDT": mpnn_ss_plddt,
                        "Unrelaxed_Clashes": num_clashes_mpnn,
                        "Relaxed_Clashes": num_clashes_mpnn_relaxed,
                        "Binder_Energy_Score": mpnn_interface_scores["binder_score"],
                        "Surface_Hydrophobicity": mpnn_interface_scores[
                            "surface_hydrophobicity"
                        ],
                        "ShapeComplementarity": mpnn_interface_scores["interface_sc"],
                        "PackStat": mpnn_interface_scores["interface_packstat"],
                        "dG": mpnn_interface_scores["interface_dG"],
                        "dSASA": mpnn_interface_scores["interface_dSASA"],
                        "dG/dSASA": mpnn_interface_scores["interface_dG_SASA_ratio"],
                        "Interface_SASA_%": mpnn_interface_scores["interface_fraction"],
                        "Interface_Hydrophobicity": mpnn_interface_scores[
                            "interface_hydrophobicity"
                        ],
                        "n_InterfaceResidues": mpnn_interface_scores["interface_nres"],
                        "n_InterfaceHbonds": mpnn_interface_scores[
                            "interface_interface_hbonds"
                        ],
                        "InterfaceHbondsPercentage": mpnn_interface_scores[
                            "interface_hbond_percentage"
                        ],
                        "n_InterfaceUnsatHbonds": mpnn_interface_scores[
                            "interface_delta_unsat_hbonds"
                        ],
                        "InterfaceUnsatHbondsPercentage": mpnn_interface_scores[
                            "interface_delta_unsat_hbonds_percentage"
                        ],
                        "InterfaceAAs": mpnn_interface_AA,
                        "Interface_Helix%": mpnn_alpha_interface,
                        "Interface_BetaSheet%": mpnn_beta_interface,
                        "Interface_Loop%": mpnn_loops_interface,
                        "Binder_Helix%": mpnn_alpha,
                        "Binder_BetaSheet%": mpnn_beta,
                        "Binder_Loop%": mpnn_loops,
                        "Hotspot_RMSD": rmsd_site,
                        "Target_RMSD": target_rmsd,
                    }
                )

                # save space by removing unrelaxed predicted mpnn complex pdb?
                if advanced_settings["remove_unrelaxed_complex"]:
                    os.remove(mpnn_design_pdb)

        # calculate complex averages
        mpnn_complex_averages = calculate_averages(
            mpnn_complex_statistics, handle_aa=True
        )

        ### Predict binder alone in single sequence mode
        binder_statistics = predict_binder_alone(
            binder_prediction_model,
            mpnn_sequence["seq"],
            mpnn_design_name,
            traj_data.Length,
            trajectory_pdb,
            binder_chain,
            prediction_models,
            advanced_settings,
            design_paths,
        )

        # extract RMSDs of binder to the original trajectory
        for model_num in prediction_models:
            mpnn_binder_pdb = os.path.join(
                design_paths["MPNN/Binder"],
                f"{mpnn_design_name}_model{model_num+1}.pdb",
            )

            if os.path.exists(mpnn_binder_pdb):
                rmsd_binder = unaligned_rmsd(
                    trajectory_pdb, mpnn_binder_pdb, binder_chain, "A"
                )

            # append to statistics
            binder_statistics[model_num + 1].update({"Binder_RMSD": rmsd_binder})

            # save space by removing binder monomer models?
            if advanced_settings["remove_binder_monomer"]:
                os.remove(mpnn_binder_pdb)

        # calculate binder averages
        binder_averages = calculate_averages(binder_statistics)

        # analyze sequence to make sure there are no cysteins and it contains residues that absorb UV for detection
        seq_notes = validate_design_sequence(
            mpnn_sequence["seq"],
            mpnn_complex_averages.get("Relaxed_Clashes", None),
            advanced_settings,
        )

        # measure time to generate design
        mpnn_end_time = time.time() - mpnn_time
        elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"

        # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
        model_numbers = range(1, 6)
        statistics_labels = [
            "pLDDT",
            "pTM",
            "i_pTM",
            "pAE",
            "i_pAE",
            "i_pLDDT",
            "ss_pLDDT",
            "Unrelaxed_Clashes",
            "Relaxed_Clashes",
            "Binder_Energy_Score",
            "Surface_Hydrophobicity",
            "ShapeComplementarity",
            "PackStat",
            "dG",
            "dSASA",
            "dG/dSASA",
            "Interface_SASA_%",
            "Interface_Hydrophobicity",
            "n_InterfaceResidues",
            "n_InterfaceHbonds",
            "InterfaceHbondsPercentage",
            "n_InterfaceUnsatHbonds",
            "InterfaceUnsatHbondsPercentage",
            "Interface_Helix%",
            "Interface_BetaSheet%",
            "Interface_Loop%",
            "Binder_Helix%",
            "Binder_BetaSheet%",
            "Binder_Loop%",
            "InterfaceAAs",
            "Hotspot_RMSD",
            "Target_RMSD",
        ]

        # Initialize mpnn_data with the non-statistical data
        mpnn_data = [
            mpnn_design_name,
            advanced_settings["design_algorithm"],
            traj_data.Length,
            traj_data.Seed,
            traj_data.Helicity,
            target_settings["target_hotspot_residues"],
            mpnn_sequence["seq"],
            mpnn_interface_residues,
            mpnn_score,
            mpnn_seqid,
        ]

        # Add the statistical data for mpnn_complex
        for label in statistics_labels:
            mpnn_data.append(mpnn_complex_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(
                    mpnn_complex_statistics.get(model, {}).get(label, None)
                )

        # Add the statistical data for binder
        for label in [
            "pLDDT",
            "pTM",
            "pAE",
            "Binder_RMSD",
        ]:  # These are the labels for binder alone
            mpnn_data.append(binder_averages.get(label, None))
            for model in model_numbers:
                mpnn_data.append(binder_statistics.get(model, {}).get(label, None))

        # Add the remaining non-statistical data
        mpnn_data.extend(
            [
                elapsed_mpnn_text,
                seq_notes,
                traj_data.TargetSettings,
                traj_data.Filters,
                traj_data.AdvancedSettings,
            ]
        )

        # insert data into csv
        insert_data(mpnn_csv, mpnn_data)

        # find best model number by pLDDT
        plddt_values: dict[int, float] = {
            i: mpnn_data[i] for i in range(11, 15) if mpnn_data[i] is not None
        }

        # Find the key with the highest value
        highest_plddt_key = int(max(plddt_values, key=plddt_values.get))

        # Output the number part of the key
        best_model_number = highest_plddt_key - 10
        best_model_pdb = os.path.join(
            design_paths["MPNN/Relaxed"],
            f"{mpnn_design_name}_model{best_model_number}.pdb",
        )

        # run design data against filter thresholds
        filter_conditions = check_filters(mpnn_data, mpnn_csv_labels, filter_settings)
        if not isinstance(filter_conditions, list):
            print(mpnn_design_name + " passed all filters")
            accepted_mpnn += 1

            # copy designs to accepted folder
            shutil.copy2(best_model_pdb, design_paths["Accepted"])

            # insert data into final csv
            final_data = [""] + mpnn_data
            insert_data(final_csv, final_data)

            # copy animation from accepted trajectory
            if advanced_settings["save_design_animations"]:
                accepted_animation = os.path.join(
                    design_paths["Accepted/Animation"],
                    f"{traj_data.Design}.html",
                )
                if not os.path.exists(accepted_animation):
                    shutil.copy2(
                        os.path.join(
                            design_paths["Trajectory/Animation"],
                            f"{traj_data.Design}.html",
                        ),
                        accepted_animation,
                    )

            # copy plots of accepted trajectory
            plot_files = os.listdir(design_paths["Trajectory/Plots"])
            plots_to_copy = [
                f
                for f in plot_files
                if f.startswith(traj_data.Design) and f.endswith(".png")
            ]
            for accepted_plot in plots_to_copy:
                source_plot = os.path.join(
                    design_paths["Trajectory/Plots"], accepted_plot
                )
                target_plot = os.path.join(
                    design_paths["Accepted/Plots"], accepted_plot
                )
                if not os.path.exists(target_plot):
                    shutil.copyfile(source_plot, target_plot)

        else:
            print(f"Unmet filter conditions for {mpnn_design_name}")
            failure_df = pd.read_csv(failure_csv)
            special_prefixes = (
                "Average_",
                "1_",
                "2_",
                "3_",
                "4_",
                "5_",
            )
            incremented_columns = set()

            for column in filter_conditions:
                base_column = column
                for prefix in special_prefixes:
                    if column.startswith(prefix):
                        base_column = column.split("_", 1)[1]

                if base_column not in incremented_columns:
                    failure_df[base_column] = failure_df[base_column] + 1
                    incremented_columns.add(base_column)

            failure_df.to_csv(failure_csv, index=False)
            shutil.copy(best_model_pdb, design_paths["Rejected"])

        # increase MPNN design number
        mpnn_n += 1

        # if enough mpnn sequences of the same trajectory pass filters then stop
        if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
            break

    if accepted_mpnn >= 1:
        print("Found " + str(accepted_mpnn) + " MPNN designs passing filters")
    else:
        print("No accepted MPNN designs found for this trajectory.")

    # save space by removing unrelaxed design trajectory PDB
    if advanced_settings["remove_unrelaxed_trajectory"]:
        os.remove(trajectory_pdb)

    # measure time it took to generate designs for one trajectory
    design_time = time.time() - design_start_time
    design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
    print(
        "Design and validation of trajectory "
        + traj_data.Design
        + " took: "
        + design_time_text
    )
    return accepted_mpnn


def init_design_trajectory(
    design_models: list[int],
    design_paths: dict[str, str],
    target_settings: dict[str, Any],
    advanced_settings: dict[str, Any],
    failure_csv: str,
    basic_settings_filename: str,
    filters_filename: str,
    advanced_settings_filename: str,
    binder_chain: str = "B",
):
    """Main function to start the design trajectory for a new binder.

    Args:
        design_models: List of model numbers to use for ColabDesign.
        design_paths: Dictionary containing paths for saving designs and statistics.
        target_settings: Dictionary containing target-specific settings.
        advanced_settings: Dictionary containing advanced design settings.
        failure_csv: Path to CSV file for logging failure statistics.
        basic_settings_filename: Name of the target-specific settings file
        filters_filename: Name of the filters configuration file.
        advanced_settings_filename: Name of the advanced settings file.
        binder_chain: Chain identifier for the binder. This is a placeholder in case multi-chain parsing in ColabDesign gets changed.

    Returns:
        List containing trajectory data in the order defined in
        `functions.generic_utils.TrajectoryData`.
    """
    ### Initialise design
    # measure time to generate design
    trajectory_start_time = time.time()

    # generate random seed to vary designs
    rng = np.random.default_rng()
    seed = rng.integers(low=0, high=999999)

    # sample binder design length randomly from defined distribution
    samples = np.arange(
        min(target_settings["lengths"]), max(target_settings["lengths"]) + 1
    )
    length = np.random.choice(samples)

    # load desired helicity value to sample different secondary structure contents
    helicity_value = load_helicity(advanced_settings)

    # generate design name and check if same trajectory was already run
    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s" + str(seed)
    trajectory_dirs = [
        "Trajectory",
        "Trajectory/Relaxed",
        "Trajectory/LowConfidence",
        "Trajectory/Clashing",
    ]
    trajectory_exists = any(
        os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb"))
        for trajectory_dir in trajectory_dirs
    )
    if trajectory_exists:
        return  # TODO: add a counter

    print("Starting trajectory: " + design_name)

    ### Begin binder hallucination
    trajectory = binder_hallucination(
        design_name,
        target_settings["starting_pdb"],
        target_settings["chains"],
        target_settings["target_hotspot_residues"],
        length,
        seed,
        helicity_value,
        design_models,
        advanced_settings,
        design_paths,
        failure_csv,
    )
    trajectory_metrics = copy_dict(
        trajectory.aux["log"]
    )  # contains plddt, ptm, i_ptm, pae, i_pae
    trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")

    # round the metrics to two decimal places
    trajectory_metrics = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in trajectory_metrics.items()
    }

    # time trajectory
    trajectory_time = time.time() - trajectory_start_time
    trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
    print(f"Starting trajectory took: {trajectory_time_text}\n")

    # Proceed if there is no trajectory termination signal
    if trajectory_metrics["terminate"] != "":
        return  # TODO: add a counter

    # Relax binder to calculate statistics
    trajectory_relaxed = os.path.join(
        design_paths["Trajectory/Relaxed"], design_name + ".pdb"
    )
    pr_relax(trajectory_pdb, trajectory_relaxed)

    # Calculate clashes before and after relaxation
    num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
    num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

    # secondary structure content of starting trajectory binder and interface
    (
        trajectory_alpha,
        trajectory_beta,
        trajectory_loops,
        trajectory_alpha_interface,
        trajectory_beta_interface,
        trajectory_loops_interface,
        trajectory_i_plddt,
        trajectory_ss_plddt,
    ) = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

    # analyze interface scores for relaxed af2 trajectory
    (
        trajectory_interface_scores,
        trajectory_interface_AA,
        trajectory_interface_residues,
    ) = score_interface(trajectory_relaxed, binder_chain)

    # starting binder sequence
    trajectory_sequence = trajectory.get_seq(get_best=True)[0]

    # analyze sequence
    traj_seq_notes = validate_design_sequence(
        trajectory_sequence, num_clashes_relaxed, advanced_settings
    )

    # target structure RMSD compared to input PDB
    trajectory_target_rmsd = target_pdb_rmsd(
        trajectory_pdb,
        target_settings["starting_pdb"],
        target_settings["chains"],
    )

    # save trajectory statistics into CSV
    trajectory_data = [
        design_name,
        advanced_settings["design_algorithm"],
        length,
        seed,
        helicity_value,
        target_settings["target_hotspot_residues"],
        trajectory_sequence,
        trajectory_interface_residues,
        trajectory_metrics["plddt"],
        trajectory_metrics["ptm"],
        trajectory_metrics["i_ptm"],
        trajectory_metrics["pae"],
        trajectory_metrics["i_pae"],
        trajectory_i_plddt,
        trajectory_ss_plddt,
        num_clashes_trajectory,
        num_clashes_relaxed,
        trajectory_interface_scores["binder_score"],
        trajectory_interface_scores["surface_hydrophobicity"],
        trajectory_interface_scores["interface_sc"],
        trajectory_interface_scores["interface_packstat"],
        trajectory_interface_scores["interface_dG"],
        trajectory_interface_scores["interface_dSASA"],
        trajectory_interface_scores["interface_dG_SASA_ratio"],
        trajectory_interface_scores["interface_fraction"],
        trajectory_interface_scores["interface_hydrophobicity"],
        trajectory_interface_scores["interface_nres"],
        trajectory_interface_scores["interface_interface_hbonds"],
        trajectory_interface_scores["interface_hbond_percentage"],
        trajectory_interface_scores["interface_delta_unsat_hbonds"],
        trajectory_interface_scores["interface_delta_unsat_hbonds_percentage"],
        trajectory_alpha_interface,
        trajectory_beta_interface,
        trajectory_loops_interface,
        trajectory_alpha,
        trajectory_beta,
        trajectory_loops,
        trajectory_interface_AA,
        trajectory_target_rmsd,
        trajectory_time_text,
        traj_seq_notes,
        basic_settings_filename,
        filters_filename,
        advanced_settings_filename,
    ]
    return trajectory_data


if __name__ == "__main__":
    # Check if JAX-capable GPU is available, otherwise exit
    check_jax_gpu()

    # perform checks of input setting files
    args = parse_input_paths()
    settings_path, filters_path, advanced_path = perform_input_check(args)

    ### load settings from JSON
    target_settings, advanced_settings, filter_settings = load_json_settings(
        settings_path, filters_path, advanced_path
    )

    settings_file = os.path.basename(settings_path).split(".")[0]
    filters_file = os.path.basename(filters_path).split(".")[0]
    advanced_file = os.path.basename(advanced_path).split(".")[0]

    ### load AF2 model settings
    design_models, prediction_models, multimer_validation = load_af2_models(
        advanced_settings["use_multimer_design"]
    )

    ### set package settings
    bindcraft_folder = os.path.dirname(os.path.realpath(__file__))
    advanced_settings = perform_advanced_settings_check(
        advanced_settings, bindcraft_folder
    )

    ### generate directories, design path names can be found within the function
    design_paths = generate_directories(target_settings["design_path"])

    ### generate dataframes
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(
        target_settings["design_path"], "trajectory_stats.csv"
    )
    mpnn_csv = os.path.join(target_settings["design_path"], "mpnn_design_stats.csv")
    final_csv = os.path.join(target_settings["design_path"], "final_design_stats.csv")
    failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args.filters)

    ####################################
    ####################################
    ####################################
    ### initialise PyRosetta
    pr.init(
        f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1'
    )

    ####################################
    # initialise counters
    script_start_time = time.time()
    trajectory_n = 1
    accepted_designs = 0

    ### start design loop
    while True:
        ### check if we have the target number of binders
        final_designs_reached = check_accepted_designs(
            design_paths,
            mpnn_csv,
            final_labels,
            final_csv,
            advanced_settings,
            target_settings,
            design_labels,
        )
        if final_designs_reached:
            # stop design loop execution
            break

        ### check if we reached maximum allowed trajectories
        max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)
        if max_trajectories_reached:
            break

        ### Initialise design
        # measure time to generate design
        trajectory_start_time = time.time()

        trajectory_data = init_design_trajectory(
            design_models,
            design_paths,
            target_settings,
            advanced_settings,
            failure_csv,
            settings_file,
            filters_file,
            advanced_file,
        )
        if not trajectory_data:
            continue
        insert_data(trajectory_csv, trajectory_data)

        if advanced_settings["enable_mpnn"]:
            traj_data = TrajectoryData(*trajectory_data)
            num_mpnn_designs = mpnn_optimize_trajectory(
                traj_data,
                prediction_models,
                design_paths,
                target_settings,
                advanced_settings,
                filter_settings,
                mpnn_csv,
                design_labels,
                failure_csv,
                final_csv,
                multimer_validation,
            )
            accepted_designs += num_mpnn_designs

        # analyse the rejection rate of trajectories to see if we need to readjust the design weights
        if (
            trajectory_n >= advanced_settings["start_monitoring"]
            and advanced_settings["enable_rejection_check"]
        ):
            acceptance = accepted_designs / trajectory_n
            if not acceptance >= advanced_settings["acceptance_rate"]:
                print(
                    "The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!"
                )
                print("Script execution stopping...")
                break

        # increase trajectory number
        trajectory_n += 1

    ### Script finished
    elapsed_time = time.time() - script_start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
    print(
        "Finished all designs. Script execution for "
        + str(trajectory_n)
        + " trajectories took: "
        + elapsed_text
    )
