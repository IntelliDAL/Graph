
# In order to ensure the fairness and comparability of the test results, 
# we directly copied and followed the dataset downloading and preprocessing steps 
# from other recent works with good impact.

# At the same time, in order not to cause unnecessary misunderstandings 
# during the blind review process and to respect the work of others, 
# we append URLs that link to their dataset downloads and processing.


def load_graph_dataset(dataset_name):
    # Directly copy from https://github.com/THUDM/GraphMAE/blob/main/graphmae/datasets/data_util.py
    "..."

    graph = "..."
    feature_dim = "..."
    num_classes = "..."

    return graph, (feature_dim, num_classes)


# We use DGL, however, AD-GCL uses PyTorch Geometric. 
# So we copy codes from https://github.com/susheels/adgcl/tree/main/datasets, modified the code in DGL version.
# Here we place the modified code, other codes can be directly copied from the URL.
import dgl
import torch
import numpy as np
from rdkit import Chem

def mol_to_dgl_graph(mol):
    """
    Converts an RDKit molecule object into a DGL Graph. Assumes the use of simplified atom and bond features.
    
    Args:
        mol (rdkit.Mol): The molecule to convert.
    
    Returns:
        dgl.DGLGraph: A DGL Graph object with node and edge features.
    """
    # Node features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
        ]
        atom_features_list.append(atom_feature)
    node_feats = torch.tensor(atom_features_list, dtype=torch.long)
    
    # Edge features
    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        src.append(bond.GetBeginAtomIdx())
        dst.append(bond.GetEndAtomIdx())
        edge_feature = [
            allowable_features['possible_bonds'].index(bond.GetBondType()),
            allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
        ]
        edge_feats.append(edge_feature)
        # Add edges in both directions
        src.append(bond.GetEndAtomIdx())
        dst.append(bond.GetBeginAtomIdx())
        edge_feats.append(edge_feature)

    edge_feats = torch.tensor(edge_feats, dtype=torch.long)
    # Create DGL graph
    g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    g.ndata['feat'] = node_feats
    g.edata['feat'] = edge_feats
    
    return g


def load_regression_dataset(dataset_name):

    graph = mol_to_dgl_graph(dataset_name)
    feature_dim = graph.ndata["feat"].shape[1]
    num_classes = 2 # Manual predefinition is convenient.

    return graph, (feature_dim, num_classes)

