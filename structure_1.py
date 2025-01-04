import requests
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import biotite.structure as struc
import biotite.structure.io as strucio

class ProteinStructureMiner:
    def __init__(self, protein_name: str, uniprot_ids: Dict[str, str]):
        """
        Initialize the ProteinStructureMiner with protein details
        
        :param protein_name: Name of the protein to analyze
        :param uniprot_ids: Dictionary of UniProt IDs for different species
        """
        self.protein_name = protein_name
        self.uniprot_ids = uniprot_ids
        self.alphafold_structures = {}

    def fetch_alphafold_structures(self):
        """
        Fetch protein structures from AlphaFold database
        
        :return: Dictionary of structure files for each species
        """
        structures = {}
        
        # Direct download URLs for AlphaFold structures
        base_url = "https://alphafold.ebi.ac.uk/files"
        
        for species, uniprot_id in self.uniprot_ids.items():
            try:
                # Construct filename based on species-specific UniProtID
                filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
                full_url = f"{base_url}/{filename}"
                
                # Download PDB file
                response = requests.get(full_url)
                
                if response.status_code == 200:
                    # Save PDB file
                    local_filename = f"{species.replace(' ', '_')}_{self.protein_name}_structure.pdb"
                    with open(local_filename, 'wb') as f:
                        f.write(response.content)
                    
                    structures[species] = local_filename
                    print(f"Successfully downloaded structure for {species}")
                else:
                    print(f"No structure found for {species} (Status code: {response.status_code})")
            
            except requests.exceptions.RequestException as e:
                print(f"Error fetching structure for {species}: {e}")
            except Exception as e:
                print(f"Unexpected error for {species}: {e}")
        
        if not structures:
            raise ValueError("No protein structures could be retrieved")
        
        self.alphafold_structures = structures
        return structures

    def structural_alignment(self, structures: Dict[str, str]):
        """
        Perform basic structural comparison of protein structures
        
        :param structures: Dictionary of structure file paths
        :return: Alignment metrics
        """
        try:
            # Read structures
            structure_data = {}
            for species, filepath in structures.items():
                try:
                    structure = strucio.load_structure(filepath)
                    structure_data[species] = structure
                except Exception as e:
                    print(f"Error reading structure for {species}: {e}")
            
            if not structure_data:
                raise ValueError("No valid structures to align")
            
            # Compute pairwise RMSD
            species_list = list(structure_data.keys())
            rmsd_matrix = np.zeros((len(species_list), len(species_list)))
            
            for i, species1 in enumerate(species_list):
                for j, species2 in enumerate(species_list):
                    if i != j:
                        rmsd_matrix[i, j] = self._compute_rmsd(
                            structure_data[species1], 
                            structure_data[species2]
                        )
            
            # Visualize RMSD
            plt.figure(figsize=(10, 8))
            plt.imshow(rmsd_matrix, cmap='coolwarm')
            plt.colorbar(label='RMSD')
            plt.xticks(range(len(species_list)), species_list, rotation=45)
            plt.yticks(range(len(species_list)), species_list)
            plt.title(f'Structural Similarity of {self.protein_name} Protein')
            plt.tight_layout()
            plt.savefig(f'{self.protein_name}_structural_similarity.png')
            plt.close()
            
            return {
                'rmsd_matrix': rmsd_matrix,
                'species': species_list
            }
        
        except Exception as e:
            print(f"Structural alignment error: {e}")
            return None

    def _compute_rmsd(self, structure1: Any, structure2: Any) -> float:
        """
        Compute Root Mean Square Deviation between two structures
        
        :param structure1: First protein structure
        :param structure2: Second protein structure
        :return: RMSD value
        """
        try:
            # Extract alpha carbon coordinates
            ca_mask1 = structure1.atom_name == "CA"
            ca_mask2 = structure2.atom_name == "CA"
            
            coords1 = structure1.coord[ca_mask1]
            coords2 = structure2.coord[ca_mask2]
            
            # Ensure same length for comparison
            min_len = min(len(coords1), len(coords2))
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]
            
            # Compute RMSD
            diff = coords1 - coords2
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            
            return rmsd
        except Exception as e:
            print(f"RMSD computation error: {e}")
            return np.nan

    def compute_structural_conservation(self, structures: Dict[str, str]):
        """
        Compute basic structural conservation metrics
        
        :param structures: Dictionary of structure file paths
        :return: Conservation metrics
        """
        try:
            # Read structures
            structure_data = {}
            for species, filepath in structures.items():
                structure = strucio.load_structure(filepath)
                structure_data[species] = structure
            
            # Extract alpha carbon information
            ca_structures = {}
            for species, structure in structure_data.items():
                ca_mask = structure.atom_name == "CA"
                ca_structures[species] = {
                    'coords': structure.coord[ca_mask],
                    'residues': structure.res_name[ca_mask]
                }
            
            # Find the minimum length across all structures
            min_length = min(len(ca_struct['coords']) for ca_struct in ca_structures.values())
            
            # Compute conservation metrics
            conservation_scores = []
            species_list = list(ca_structures.keys())
            
            # Compute spatial variation for each position
            for pos in range(min_length):
                # Collect coordinates at this position across species
                coords_at_pos = [
                    species_struct['coords'][pos] 
                    for species_struct in ca_structures.values()
                ]
                
                # Compute spatial variation
                spatial_std = np.std(coords_at_pos, axis=0)
                spatial_magnitude = np.linalg.norm(spatial_std)
                
                # Simplified conservation score
                conservation_score = 1 / (spatial_magnitude + 1)
                
                conservation_scores.append({
                    'position': pos,
                    'conservation_score': conservation_score,
                    'spatial_variation': spatial_magnitude
                })
            
            # Create DataFrame and save to Excel
            df = pd.DataFrame(conservation_scores)
            excel_filename = f'{self.protein_name}_structural_conservation.xlsx'
            df.to_excel(excel_filename, index=False)
            print(f"Conservation scores saved to {excel_filename}")
            
            # Visualize conservation scores
            plt.figure(figsize=(12, 6))
            plt.plot(df['position'], df['conservation_score'], linewidth=2)
            plt.title(f'Structural Conservation Scores for {self.protein_name}')
            plt.xlabel('Sequence Position')
            plt.ylabel('Conservation Score')
            plt.ylim(0, 1.1)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Highlight highly conserved regions
            highly_conserved = df[df['conservation_score'] > 0.8]
            plt.scatter(highly_conserved['position'], highly_conserved['conservation_score'], 
                        color='red', label='Highly Conserved (>0.8)')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.protein_name}_structural_conservation_plot.png')
            plt.close()
            
            return conservation_scores
    
        except Exception as e:
            print(f"Structural conservation computation error: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    # Provide species-specific UniProt IDs for the protein
    brca1_uniprot_ids = {
        "Homo sapiens": "P38398",     # Human
        "Mus musculus": "Q9QZF9",     # Mouse
        "Danio rerio": "Q6NZN9",      # Zebrafish
        "Pan troglodytes": "A0A2R5S4X0", # Chimpanzee
        "Canis lupus": "A0A1L8G2Q2"   # Dog
    }
    
    # Analyze BRCA1 protein
    structure_miner = ProteinStructureMiner('BRCA1', brca1_uniprot_ids)
    
    try:
        # Fetch structures
        structures = structure_miner.fetch_alphafold_structures()
        
        # Perform structural alignment
        alignment_result = structure_miner.structural_alignment(structures)
        
        # Compute structural conservation
        conservation_scores = structure_miner.compute_structural_conservation(structures)
        
        # Print detailed conservation information
        if conservation_scores:
            print("\nDetailed Structural Conservation Scores:")
            for score in conservation_scores[:10]:  # Print first 10 scores
                print(f"Position {score['position']}: "
                      f"Score = {score['conservation_score']:.4f}, "
                      f"Spatial Variation = {score['spatial_variation']:.4f}")
    
    except Exception as e:
        print(f"An error occurred during protein structure analysis: {e}")

if __name__ == '__main__':
    main()