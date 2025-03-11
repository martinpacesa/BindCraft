####################################
################## General functions
####################################
### Import dependencies
import os
import json
import jax
import shutil
import zipfile
import random
import math
import pandas as pd
import numpy as np

# Define labels for dataframes
def generate_dataframe_labels():
    # labels for trajectory
    trajectory_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes',
                        'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity', 'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues',
                        'n_InterfaceHbonds', 'InterfaceHbondsPercentage', 'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%',
                        'Binder_Helix%', 'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Target_RMSD', 'TrajectoryTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    # labels for mpnn designs
    core_labels = ['pLDDT', 'pTM', 'i_pTM', 'pAE', 'i_pAE', 'i_pLDDT', 'ss_pLDDT', 'Unrelaxed_Clashes', 'Relaxed_Clashes', 'Binder_Energy_Score', 'Surface_Hydrophobicity',
                    'ShapeComplementarity', 'PackStat', 'dG', 'dSASA', 'dG/dSASA', 'Interface_SASA_%', 'Interface_Hydrophobicity', 'n_InterfaceResidues', 'n_InterfaceHbonds', 'InterfaceHbondsPercentage',
                    'n_InterfaceUnsatHbonds', 'InterfaceUnsatHbondsPercentage', 'Interface_Helix%', 'Interface_BetaSheet%', 'Interface_Loop%', 'Binder_Helix%', 
                    'Binder_BetaSheet%', 'Binder_Loop%', 'InterfaceAAs', 'Hotspot_RMSD', 'Target_RMSD', 'Binder_pLDDT', 'Binder_pTM', 'Binder_pAE', 'Binder_RMSD']

    design_labels = ['Design', 'Protocol', 'Length', 'Seed', 'Helicity', 'Target_Hotspot', 'Sequence', 'InterfaceResidues', 'MPNN_score', 'MPNN_seq_recovery']

    for label in core_labels:
        design_labels += ['Average_' + label] + [f'{i}_{label}' for i in range(1, 6)]

    design_labels += ['DesignTime', 'Notes', 'TargetSettings', 'Filters', 'AdvancedSettings']

    final_labels = ['Rank'] + design_labels

    return trajectory_labels, design_labels, final_labels

# Create base directions of the project
def generate_directories(design_path):
    design_path_names = ["Accepted", "Accepted/Ranked", "Accepted/Animation", "Accepted/Plots", "Accepted/Pickle", "Trajectory",
                        "Trajectory/Relaxed", "Trajectory/Plots", "Trajectory/Clashing", "Trajectory/LowConfidence", "Trajectory/Animation",
                        "Trajectory/Pickle", "MPNN", "MPNN/Binder", "MPNN/Sequences", "MPNN/Relaxed", "Rejected"]
    design_paths = {}

    # make directories and set design_paths[FOLDER_NAME] variable
    for name in design_path_names:
        path = os.path.join(design_path, name)
        os.makedirs(path, exist_ok=True)
        design_paths[name] = path

    return design_paths

# generate CSV file for tracking designs not passing filters
def generate_filter_pass_csv(failure_csv, filter_json):
    if not os.path.exists(failure_csv):
        with open(filter_json, 'r') as file:
            data = json.load(file)
        
        # Create a list of modified keys
        names = ['Trajectory_logits_pLDDT', 'Trajectory_softmax_pLDDT', 'Trajectory_one-hot_pLDDT', 'Trajectory_final_pLDDT', 'Trajectory_Contacts', 'Trajectory_Clashes', 'Trajectory_WrongHotspot']
        special_prefixes = ('Average_', '1_', '2_', '3_', '4_', '5_')
        tracked_filters = set()

        for key in data.keys():
            processed_name = key  # Use the full key by default

            # Check if the key starts with any special prefixes
            for prefix in special_prefixes:
                if key.startswith(prefix):
                    # Strip the prefix and use the remaining part
                    processed_name = key.split('_', 1)[1]
                    break

            # Handle 'InterfaceAAs' with appending amino acids
            if 'InterfaceAAs' in processed_name:
                # Generate 20 variations of 'InterfaceAAs' with amino acids appended
                amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                for aa in amino_acids:
                    variant_name = f"InterfaceAAs_{aa}"
                    if variant_name not in tracked_filters:
                        names.append(variant_name)
                        tracked_filters.add(variant_name)
            elif processed_name not in tracked_filters:
                # Add processed name if it hasn't been added before
                names.append(processed_name)
                tracked_filters.add(processed_name)

        # make dataframe with 0s
        df = pd.DataFrame(columns=names)
        df.loc[0] = [0] * len(names)

        df.to_csv(failure_csv, index=False)

# update failure rates from trajectories and early predictions
def update_failures(failure_csv, failure_column_or_dict):
    failure_df = pd.read_csv(failure_csv)
    
    def strip_model_prefix(name):
        # Strips the model-specific prefix if it exists
        parts = name.split('_')
        if parts[0].isdigit():
            return '_'.join(parts[1:])
        return name
    
    # update dictionary coming from complex prediction
    if isinstance(failure_column_or_dict, dict):
        # Update using a dictionary of failures
        for filter_name, count in failure_column_or_dict.items():
            stripped_name = strip_model_prefix(filter_name)
            if stripped_name in failure_df.columns:
                failure_df[stripped_name] += count
            else:
                failure_df[stripped_name] = count
    else:
        # Update a single column from trajectory generation
        failure_column = strip_model_prefix(failure_column_or_dict)
        if failure_column in failure_df.columns:
            failure_df[failure_column] += 1
        else:
            failure_df[failure_column] = 1
    
    failure_df.to_csv(failure_csv, index=False)

# Check if number of trajectories generated
def check_n_trajectories(design_paths, advanced_settings):
    n_trajectories = [f for f in os.listdir(design_paths["Trajectory/Relaxed"]) if f.endswith('.pdb')]

    if advanced_settings["max_trajectories"] is not False and len(n_trajectories) >= advanced_settings["max_trajectories"]:
        print(f"Target number of {str(len(n_trajectories))} trajectories reached, stopping execution...")
        return True
    else:
        return False

# Check if we have required number of accepted targets, rank them, and analyse sequence and structure properties
def check_accepted_designs(design_paths, mpnn_csv, final_labels, final_csv, advanced_settings, target_settings, design_labels):
    accepted_binders = [f for f in os.listdir(design_paths["Accepted"]) if f.endswith('.pdb')]

    if len(accepted_binders) >= target_settings["number_of_final_designs"]:
        print(f"Target number {str(len(accepted_binders))} of designs reached! Reranking...")

        # clear the Ranked folder in case we added new designs in the meantime so we rerank them all
        for f in os.listdir(design_paths["Accepted/Ranked"]):
            os.remove(os.path.join(design_paths["Accepted/Ranked"], f))

        # load dataframe of designed binders
        design_df = pd.read_csv(mpnn_csv)
        design_df = design_df.sort_values('Average_i_pTM', ascending=False)
        
        # create final csv dataframe to copy matched rows, initialize with the column labels
        final_df = pd.DataFrame(columns=final_labels)

        # check the ranking of the designs and copy them with new ranked IDs to the folder
        rank = 1
        for _, row in design_df.iterrows():
            for binder in accepted_binders:
                target_settings["binder_name"], model = binder.rsplit('_model', 1)
                if target_settings["binder_name"] == row['Design']:
                    # rank and copy into ranked folder
                    row_data = {'Rank': rank, **{label: row[label] for label in design_labels}}
                    final_df = pd.concat([final_df, pd.DataFrame([row_data])], ignore_index=True)
                    old_path = os.path.join(design_paths["Accepted"], binder)
                    new_path = os.path.join(design_paths["Accepted/Ranked"], f"{rank}_{target_settings['binder_name']}_model{model.rsplit('.', 1)[0]}.pdb")
                    shutil.copyfile(old_path, new_path)

                    rank += 1
                    break

        # save the final_df to final_csv
        final_df.to_csv(final_csv, index=False)

        # zip large folders to save space
        if advanced_settings["zip_animations"]:
            zip_and_empty_folder(design_paths["Trajectory/Animation"], '.html')

        if advanced_settings["zip_plots"]:
            zip_and_empty_folder(design_paths["Trajectory/Plots"], '.png')

        return True

    else:
        return False

# Load required helicity value
def load_helicity(advanced_settings):
    if advanced_settings["random_helicity"] is True:
        # will sample a random bias towards helicity
        helicity_value = round(np.random.uniform(-3, 1),2)
    elif advanced_settings["weights_helicity"] != 0:
        # using a preset helicity bias
        helicity_value = advanced_settings["weights_helicity"]
    else:
        # no bias towards helicity
        helicity_value = 0
    return helicity_value

# Report JAX-capable devices
def check_jax_gpu():
    devices = jax.devices()

    has_gpu = any(device.platform == 'gpu' for device in devices)

    if not has_gpu:
        print("No GPU device found, terminating.")
        exit()
    else:
        print("Available GPUs:")
        for i, device in enumerate(devices):
            print(f"{device.device_kind}{i + 1}: {device.platform}")

# check all input files being passed
def perform_input_check(args):
    # Get the directory of the current script
    binder_script_path = os.path.dirname(os.path.abspath(__file__))

    # Ensure settings file is provided
    if not args.settings:
        print("Error: --settings is required.")
        exit()

    # Set default filters.json path if not provided
    if not args.filters:
        args.filters = os.path.join(binder_script_path, 'settings_filters', 'default_filters.json')

    # Set a random advanced json settings file if not provided
    if not args.advanced:
        args.advanced = os.path.join(binder_script_path, 'settings_advanced', 'default_4stage_multimer.json')

    return args.settings, args.filters, args.advanced

# check specific advanced settings
def perform_advanced_settings_check(advanced_settings, bindcraft_folder):
    # set paths to model weights and executables
    if bindcraft_folder == "colab":
        advanced_settings["af_params_dir"] = '/content/bindcraft/params/'
        advanced_settings["dssp_path"] = '/content/bindcraft/functions/dssp'
        advanced_settings["dalphaball_path"] = '/content/bindcraft/functions/DAlphaBall.gcc'
    else:
        # Set paths individually if they are not already set
        if not advanced_settings["af_params_dir"]:
            advanced_settings["af_params_dir"] = bindcraft_folder
        if not advanced_settings["dssp_path"]:
            advanced_settings["dssp_path"] = os.path.join(bindcraft_folder, 'functions', 'dssp')
        if not advanced_settings["dalphaball_path"]:
            advanced_settings["dalphaball_path"] = os.path.join(bindcraft_folder, 'functions', 'DAlphaBall.gcc')

    # check formatting of omit_AAs setting
        omit_aas = advanced_settings["omit_AAs"]
    if advanced_settings["omit_AAs"] in [None, False, '']:
        advanced_settings["omit_AAs"] = None
    elif isinstance(advanced_settings["omit_AAs"], str):
        advanced_settings["omit_AAs"] = advanced_settings["omit_AAs"].strip()

    return advanced_settings

# Load settings from JSONs
def load_json_settings(settings_json, filters_json, advanced_json):
    # load settings from json files
    with open(settings_json, 'r') as file:
        target_settings = json.load(file)

    with open(advanced_json, 'r') as file:
        advanced_settings = json.load(file)

    with open(filters_json, 'r') as file:
        filters = json.load(file)

    return target_settings, advanced_settings, filters

# AF2 model settings, make sure non-overlapping models with template option are being used for design and re-prediction
def load_af2_models(af_multimer_setting):
    if af_multimer_setting:
        design_models = [0,1,2,3,4]
        prediction_models = [0,1]
        multimer_validation = False
    else:
        design_models = [0,1]
        prediction_models = [0,1,2,3,4]
        multimer_validation = True

    return design_models, prediction_models, multimer_validation

# create csv for insertion of data
def create_dataframe(csv_file, columns):
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_file, index=False)

# insert row of statistics into csv
def insert_data(csv_file, data_array):
    df = pd.DataFrame([data_array])
    df.to_csv(csv_file, mode='a', header=False, index=False)

# save generated sequence
def save_fasta(design_name, sequence, design_paths):
    fasta_path = os.path.join(design_paths["MPNN/Sequences"], design_name+".fasta")
    with open(fasta_path,"w") as fasta:
        line = f'>{design_name}\n{sequence}'
        fasta.write(line+"\n")

# clean unnecessary rosetta information from PDB
def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK'))]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)

def zip_and_empty_folder(folder_path, extension):
    folder_basename = os.path.basename(folder_path)
    zip_filename = os.path.join(os.path.dirname(folder_path), folder_basename + '.zip')

    # Open the zip file in 'a' mode to append if it exists, otherwise create a new one
    with zipfile.ZipFile(zip_filename, 'a', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(folder_path):
            if file.endswith(extension):
                # Create an absolute path
                file_path = os.path.join(folder_path, file)
                # Add file to zip file, replacing it if it already exists
                zipf.write(file_path, arcname=file)
                # Remove the file after adding it to the zip
                os.remove(file_path)
    print(f"Files in folder '{folder_path}' have been zipped and removed.")

# calculate averages for statistics
def calculate_averages(statistics, handle_aa=False):
    # Initialize a dictionary to hold the sums of each statistic
    sums = {}
    # Initialize a dictionary to hold the sums of each amino acid count
    aa_sums = {}

    # Iterate over the model numbers
    for model_num in range(1, 6):  # assumes models are numbered 1 through 5
        # Check if the model's data exists
        if model_num in statistics:
            # Get the model's statistics
            model_stats = statistics[model_num]
            # For each statistic, add its value to the sum
            for stat, value in model_stats.items():
                # If this is the first time we've seen this statistic, initialize its sum to 0
                if stat not in sums:
                    sums[stat] = 0

                if value is None:
                    value = 0

                # If the statistic is mpnn_interface_AA and we're supposed to handle it separately, do so
                if handle_aa and stat == 'InterfaceAAs':
                    for aa, count in value.items():
                        # If this is the first time we've seen this amino acid, initialize its sum to 0
                        if aa not in aa_sums:
                            aa_sums[aa] = 0
                        aa_sums[aa] += count
                else:
                    sums[stat] += value

    # Now that we have the sums, we can calculate the averages
    averages = {stat: round(total / len(statistics), 2) for stat, total in sums.items()}

    # If we're handling aa counts, calculate their averages
    if handle_aa:
        aa_averages = {aa: round(total / len(statistics),2) for aa, total in aa_sums.items()}
        averages['InterfaceAAs'] = aa_averages

    return averages

# filter designs based on feature thresholds
def check_filters(mpnn_data, design_labels, filters):
    # check mpnn_data against labels
    mpnn_dict = {label: value for label, value in zip(design_labels, mpnn_data)}

    unmet_conditions = []

    # check filters against thresholds
    for label, conditions in filters.items():
        # special conditions for interface amino acid counts
        if label == 'Average_InterfaceAAs' or label == '1_InterfaceAAs' or label == '2_InterfaceAAs' or label == '3_InterfaceAAs' or label == '4_InterfaceAAs' or label == '5_InterfaceAAs':
            for aa, aa_conditions in conditions.items():
                if mpnn_dict.get(label) is None:
                    continue
                value = mpnn_dict.get(label).get(aa)
                if value is None or aa_conditions["threshold"] is None:
                    continue
                if aa_conditions["higher"]:
                    if value < aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
                else:
                    if value > aa_conditions["threshold"]:
                        unmet_conditions.append(f"{label}_{aa}")
        else:
            # if no threshold, then skip
            value = mpnn_dict.get(label)
            if value is None or conditions["threshold"] is None:
                continue
            if conditions["higher"]:
                if value < conditions["threshold"]:
                    unmet_conditions.append(label)
            else:
                if value > conditions["threshold"]:
                    unmet_conditions.append(label)

    # if all filters are passed then return True
    if len(unmet_conditions) == 0:
        return True
    # if some filters were unmet, print them out
    else:
        return unmet_conditions
