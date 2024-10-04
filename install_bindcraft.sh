#!/bin/bash
################## BindCraft installation script
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or $pkg_manager
# Default value for pkg_manager
pkg_manager='conda'

# Define the short and long options
OPTIONS=p:
LONGOPTIONS=pkg_manager:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths
install_dir=$(pwd)

### BindCraft install
printf "Installing BindCraft environment\n"
$pkg_manager create --name BindCraft python=3.9 -y
conda activate BindCraft

# install helpful packages
$pkg_manager install pandas numpy biopython==1.79 scipy"<1.13.0" pdbfixer seaborn tqdm jupyter ffmpeg -y

# install ColabDesign
pip install git+https://github.com/sokrypton/ColabDesign.git
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.htm
pip install matplotlib==3.7.1

# install PyRosetta
$pkg_manager install pyrosetta --channel https://conda.graylab.jhu.edu -y

# Download AlphaFold2 weights
mkdir -p ${install_dir}/params/
cd ${install_dir}/params/
wget -P ${install_dir}/params/ https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xvf ${install_dir}/params/alphafold_params_2022-12-06.tar
rm ${install_dir}/params/alphafold_params_2022-12-06.tar

# chmod executables
chmod +x ${install_dir}/functions/dssp
chmod +x ${install_dir}/functions/DAlphaBall.gcc

# finish
conda deactivate
printf "BindCraft environment installed\n"

############################################################################################################
############################################################################################################
################## cleanup
printf "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
printf "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
printf "Finished setting up BindCraft environment\n"
printf "Activate environment using command: \"conda activate BindCraft\""
printf "\n"
printf "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
