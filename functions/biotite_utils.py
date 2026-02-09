####################################
################ Biotite functions
####################################
### Import dependencies
from biotite.application.application import AppState, requires_state
from biotite.application.localapp import get_version, cleanup_tempfile, LocalApp
from scipy.spatial import KDTree
from subprocess import SubprocessError
from tempfile import NamedTemporaryFile
import biotite.application.dssp as b_dssp
import biotite.sequence as b_sequence
import biotite.structure as b_structure
import fastpdb
import numpy as np

# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []
    bseq = b_sequence.ProteinSequence(sequence)

    # Check if protein contains clashes after relaxation
    if num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('Contains: '+restricted_AA+'!')

    # Analyze the protein

    # Calculate the reduced extinction coefficient per 1% solution
    num_aa = bseq.get_symbol_frequency()
    extinction_coefficient_reduced = num_aa["W"] * 5500 + num_aa["Y"] * 1490
    molecular_weight = np.round(bseq.get_molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = np.round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design.')

    # Join the notes into a single string
    notes = ' '.join(note_array)

    return notes

# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    file_trajectory = fastpdb.PDBFile.read(trajectory_pdb)
    file_starting = fastpdb.PDBFile.read(starting_pdb)

    aa_trajectory = file_trajectory.get_structure(model=1)
    aa_starting = file_starting.get_structure(model=1)

    # Extract CA atoms from starting_pdb
    chain_ids = [chain_id.strip() for chain_id in chain_ids_string.split(',')]

    aa_starting = aa_starting[(np.isin(aa_starting.chain_id, chain_ids))
                              & (aa_starting.atom_name == "CA")
                              & (aa_starting.element == "C")
                              & (b_structure.filter_canonical_amino_acids(aa_starting))]

    # Extract CA atoms from chain A in trajectory_pdb
    aa_trajectory = aa_trajectory[(aa_trajectory.chain_id == "A")
                                  & (aa_trajectory.atom_name == "CA")
                                  & (aa_trajectory.element == "C")
                                  & (b_structure.filter_canonical_amino_acids(aa_trajectory))]
    
    # Ensure that both structures have the same number of residues
    min_length = min(aa_starting.array_length(), aa_trajectory.array_length())
    aa_starting = aa_starting[:min_length]
    aa_trajectory = aa_trajectory[:min_length]
    
    # Calculate RMSD using structural alignment
    superimposed, _ = b_structure.superimpose(aa_starting, aa_trajectory)
    rmsd = b_structure.rmsd(aa_starting, superimposed)

    return np.round(rmsd, 2)

# detect C alpha clashes for deformed trajectories
def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    file_structure = fastpdb.PDBFile.read(pdb_file)
    aa_structure = file_structure.get_structure(model=1)

    atoms = aa_structure[aa_structure.element != "H"]
    if only_ca:
        atoms = atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]

    tree = KDTree(atoms.coord)
    pairs_set = tree.query_pairs(threshold)
    if not pairs_set:
        return 0

    pairs = np.array(list(pairs_set))

    atoms_i = atoms[pairs[:, 0]]
    atoms_j = atoms[pairs[:, 1]]

    # If calculating sidechain clashes, only consider clashes between different chains
    if not only_ca:
        valid_mask = atoms_i.chain_id != atoms_j.chain_id
    else:
        # Exclude clashes within the same residue or sequential residues in the
        # same chain for all atoms. So only clashes that are in different
        # chains and more than 1 residue apart will be a valid clash
        valid_mask = ((atoms_i.chain_id != atoms_j.chain_id) 
                          | (np.abs(atoms_i.res_id - atoms_j.res_id) > 1))


    return valid_mask.sum()

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    file_structure = fastpdb.PDBFile.read(trajectory_pdb)
    aa_structure = file_structure.get_structure(model=1)

    # Get the specified chain
    aa_binder = aa_structure[aa_structure.chain_id == binder_chain]

    # Get atoms and coords for the target chain
    aa_target = aa_structure[aa_structure.chain_id == "A"]

    # Build KD trees for both chains
    binder_tree = KDTree(aa_binder.coord)
    target_tree = KDTree(aa_target.coord)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    is_close = np.asarray([len(idx) > 0 for idx in pairs], dtype=bool)

    if is_close.size == 0:
        return {}

    # Filter residues with the manual dictionary
    aa_binder = aa_binder[np.isin(aa_binder.res_name, list(three_to_one_map.keys()))]

    binder_resid = aa_binder[is_close].res_id
    binder_resname = aa_binder[is_close].res_name

    interacting_residues = {int(binder_resid[i]): three_to_one_map[binder_resname[i]] for i in range(binder_resid.size)}

    return interacting_residues

class DsspFromFile(b_dssp.DsspApp):
    """
    Implementation of biotite.application.dssp.DsspApp that processes input
    files directly without the need to temporarily save the structure and
    returns the secondary structure with resid and chaind.
    """
    def __init__(self, in_file, bin_path="mkdssp"):
        LocalApp.__init__(self, bin_path)
        try:
            # The parameters have changed in version 4
            self._new_cli = get_version(bin_path)[0] >= 4
        except SubprocessError:
            # In older versions, the no version is returned with `--version`
            # -> a SubprocessError is raised
            self._new_cli = False

        self._in_file = in_file
        self._out_file = NamedTemporaryFile("r", suffix=".dssp", delete=False)

    def run(self):
        if self._new_cli:
            self.set_arguments([self._in_file, self._out_file.name])
        else:
            self.set_arguments(["-i", self._in_file, "-o", self._out_file.name])
        LocalApp.run(self)

    def evaluate(self):
        LocalApp.evaluate(self)
        lines = self._out_file.read().split("\n")
        # Index where SSE records start
        sse_start = None
        for i, line in enumerate(lines):
            if line.startswith("  #  RESIDUE AA STRUCTURE"):
                sse_start = i + 1
        if sse_start is None:
            raise ValueError("DSSP file does not contain SSE records")
        # Remove "!" for missing residues
        lines = [
            line for line in lines[sse_start:] if len(line) != 0 and line[13] != "!"
        ]
        self._sse = np.zeros(len(lines), dtype="U1")
        self._sse_chainids = np.zeros(len(lines), dtype="U1")
        self._sse_resids = np.zeros(len(lines), dtype=int)
        # Parse file for SSE letters
        for i, line in enumerate(lines):
            self._sse[i] = line[16]
            self._sse_chainids[i] = line[11]
            self._sse_resids[i] = int(line[5:10])
        self._sse[self._sse == " "] = "C"

    def clean_up(self):
        LocalApp.clean_up(self)
        cleanup_tempfile(self._out_file)

    @staticmethod
    def annotate_sse(in_file, bin_path="mkdssp"):
        app = DsspFromFile(in_file, bin_path)
        app.start()
        app.join()
        return app.get_sse()

    @requires_state(AppState.JOINED)
    def get_sse(self):
        return self._sse, self._sse_chainids, self._sse_resids



# calculate secondary structure percentage of design
def calc_ss_percentage(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    file_structure = fastpdb.PDBFile.read(pdb_file)
    aa_structure = file_structure.get_structure(model=1, extra_fields=["b_factor"]) # Consider only the first model in the structure
    aa_chain = aa_structure[aa_structure.chain_id == chain_id]

    # Calculate DSSP for the model
    dssp_ss, dssp_chainids, dssp_resids = DsspFromFile.annotate_sse(pdb_file, bin_path=advanced_settings["dssp_path"])

    chain_ss = dssp_ss[dssp_chainids == chain_id]
    chain_ss_resids = dssp_resids[dssp_chainids == chain_id]

    ss_counts = {}
    ss_interface_counts = {}

    # Count secondary structures
    ss_types, ss_type_counts = np.unique(chain_ss, return_counts=True)
    ss_counts["helix"] = ss_type_counts[np.isin(ss_types, ['H', 'G', 'I'])].sum()
    ss_counts["sheet"] = ss_type_counts[ss_types == 'E'].sum()
    ss_counts["loop"] = chain_ss.size - (ss_counts["helix"] + ss_counts["sheet"])


    interacting_residues = list(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())
    interface_ss = chain_ss[np.isin(chain_ss_resids, interacting_residues)]
    interface_ss_resids = chain_ss_resids[np.isin(chain_ss_resids, interacting_residues)]

    # Calculate nonloop pLDDTs
    chain_nonloop_resids = chain_ss_resids[np.isin(chain_ss, ['H', 'G', 'I', 'E'])]
    aa_chain_nonloop = aa_chain[np.isin(aa_chain.res_id, chain_nonloop_resids)]

    plddts_ss = b_structure.apply_residue_wise(aa_chain_nonloop, aa_chain_nonloop.b_factor, np.mean)

    # Count interface secondary structures
    ss_interface_types, ss_interface_type_counts = np.unique(interface_ss, return_counts=True)
    ss_interface_counts["helix"] = ss_interface_type_counts[np.isin(ss_interface_types, ['H', 'G', 'I'])].sum()
    ss_interface_counts["sheet"] = ss_interface_type_counts[ss_interface_types == 'E'].sum()
    ss_interface_counts["loop"] = interface_ss.size - (ss_counts["helix"] + ss_counts["sheet"])

    # Calculate interface pLDDT (only use residues returned from DSSP)
    aa_interface_chain = aa_chain[np.isin(aa_chain.res_id, interface_ss_resids)]
    plddts_interface = b_structure.apply_residue_wise(aa_interface_chain, aa_interface_chain.b_factor, np.mean)

    # Calculate percentages
    total_residues = chain_ss.size
    total_interface_residues = interface_ss.size

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    if "b_factor" not in aa_structure.get_annotation_categories():
        i_plddt = 0
        ss_plddt = 0
    else:
        i_plddt = np.round(np.mean(plddts_interface) / 100, 2)
        ss_plddt = np.round(np.mean(plddts_ss) / 100, 2)

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100,2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100,2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100,2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage
