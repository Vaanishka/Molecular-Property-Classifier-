"""
Molecular featurization: Extract chemical properties from SMILES.

BBB penetration depends on molecular properties like:
- Size (molecular weight)
- Lipophilicity (LogP - how fat-soluble)
- Polarity (H-bond donors/acceptors, polar surface area)
- Flexibility (rotatable bonds)
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


class MolecularFeaturizer:
    """Extract molecular features for BBB penetration prediction."""

    def __init__(self, use_fingerprints: bool = True, fp_size: int = 2048):
        """
        Args:
            use_fingerprints: Whether to compute molecular fingerprints
            fp_size: Size of fingerprint vector (2048 bits typical)
        """
        self.use_fingerprints = use_fingerprints
        self.fp_size = fp_size

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES string to RDKit molecule object."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None

    def get_molecular_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate molecular descriptors relevant to BBB penetration.
        
        BBB Penetration Rules (general guidelines):
        - Molecular Weight: < 450 Da (smaller = easier to cross)
        - LogP: 1-3 (lipophilicity sweet spot)
        - H-bond donors: ≤ 3 (too many = hard to cross)
        - H-bond acceptors: ≤ 8
        - TPSA: < 90 Ų (polar surface area)
        - Rotatable bonds: ≤ 10 (flexibility)
        """
        descriptors = {}

        try:
            # Size
            descriptors['MolecularWeight'] = Descriptors.MolWt(mol)
            
            # Lipophilicity (how fat-soluble)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            
            # Polarity
            descriptors['HBondDonors'] = Descriptors.NumHDonors(mol)
            descriptors['HBondAcceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)  # Topological polar surface area
            
            # Flexibility
            descriptors['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)

            # Ring structures
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)

            # Composition
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            descriptors['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
            
            # Charge
            descriptors['FormalCharge'] = Chem.GetFormalCharge(mol)

        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            # Return zeros if calculation fails
            for key in ['MolecularWeight', 'LogP', 'HBondDonors', 'HBondAcceptors',
           'TPSA', 'RotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
           'NumSaturatedRings', 'NumHeteroatoms', 'NumHeavyAtoms',
           'FormalCharge']:
                descriptors[key] = 0.0

        return descriptors

    def get_morgan_fingerprint(self, mol: Chem.Mol, radius: int = 2) -> np.ndarray:
        """
        Get Morgan fingerprint (circular fingerprint).
        
        What it is: Each bit represents presence of chemical substructures.
        Example: Bit 42 = "has benzene ring"
        Used for: Molecular similarity, sometimes as model features
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=self.fp_size)
            return np.array(fp, dtype=np.float32)
        except:
            return np.zeros(self.fp_size, dtype=np.float32)

    def featurize(self, smiles: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Complete featurization of a SMILES string.
        
        Returns:
            fingerprint: Morgan fingerprint vector (or None)
            descriptors: Dictionary of molecular properties
        """
        mol = self.smiles_to_mol(smiles)

        if mol is None:
            # Invalid SMILES - return zeros
            fp = np.zeros(self.fp_size, dtype=np.float32)
            desc = {k: 0.0 for k in [
                'MolecularWeight', 'LogP', 'HBondDonors', 'HBondAcceptors',
                'TPSA', 'RotatableBonds', 'NumAromaticRings', 'NumAliphaticRings',
                'NumSaturatedRings', 'NumHeteroatoms', 'NumHeavyAtoms',
                 'FormalCharge'
            ]}
            return fp, desc

        fp = self.get_morgan_fingerprint(mol) if self.use_fingerprints else None
        descriptors = self.get_molecular_descriptors(mol)

        return fp, descriptors

    def batch_featurize(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Featurize multiple molecules at once.
        
        Returns:
            fingerprints: Array of shape (batch_size, fp_size)
            descriptors: Array of shape (batch_size, num_descriptors)
        """
        all_fps = []
        all_descs = []

        for smiles in smiles_list:
            fp, desc = self.featurize(smiles)
            all_fps.append(fp)
            all_descs.append(list(desc.values()))

        fps_array = np.stack(all_fps)
        descs_array = np.array(all_descs, dtype=np.float32)

        return fps_array, descs_array


def check_bbb_rules(smiles: str) -> Dict[str, bool]:
    """
    Check if molecule follows known BBB penetration rules.
    
    These are empirical guidelines (not guarantees):
    - Good BBB candidates follow most/all rules
    - Bad BBB candidates violate multiple rules
    
    Returns:
        Dictionary of True/False for each rule
    """
    featurizer = MolecularFeaturizer(use_fingerprints=False)
    mol = featurizer.smiles_to_mol(smiles)

    if mol is None:
        return {key: False for key in ['valid_mw', 'valid_logp', 'valid_hbd',
                                       'valid_hba', 'valid_psa', 'overall']}

    descriptors = featurizer.get_molecular_descriptors(mol)

    rules = {
        'valid_mw': descriptors['MolecularWeight'] < 450,
        'valid_logp': 1 <= descriptors['LogP'] <= 3,
        'valid_hbd': descriptors['HBondDonors'] <= 3,
        'valid_hba': descriptors['HBondAcceptors'] <= 8,
        'valid_psa': descriptors['TPSA'] < 90,
    }

    rules['overall'] = all(rules.values())

    return rules


if __name__ == "__main__":
    featurizer = MolecularFeaturizer(use_fingerprints=True, fp_size=2048)

    test_molecules = {
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "Morphine": "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O"
    }

    print("=" * 70)
    print("Molecular Featurization Test")
    print("=" * 70)

    for name, smiles in test_molecules.items():
        print(f"\n🧪 {name}")
        print(f"   SMILES: {smiles}")

        fp, descriptors = featurizer.featurize(smiles)
        
        print(f"\n   📊 Descriptors:")
        for key in ['MolecularWeight', 'LogP', 'HBondDonors', 'HBondAcceptors', 'TPSA']:
            print(f"      {key}: {descriptors[key]:.2f}")

        print(f"\n   ✓ Fingerprint shape: {fp.shape}")
        print(f"   ✓ Non-zero bits: {np.count_nonzero(fp)}")

        rules = check_bbb_rules(smiles)
        print(f"\n   🎯 BBB Rule Compliance:")
        for rule, passes in rules.items():
            if rule != 'overall':
                symbol = "✓" if passes else "✗"
                print(f"      {symbol} {rule}: {passes}")
        print(f"      Overall: {rules['overall']}")

        print("-" * 70)

    print("=" * 70)
    print("✓ Test complete!")
    print("=" * 70)