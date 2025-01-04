import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import Entrez, AlignIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SubsMat import MatrixInfo
from typing import Dict


class ProteinDataMiner:
    def __init__(self, protein_name: str, email: str):
        self.protein_name = protein_name
        self.species_list = [
            'Homo sapiens', 'Mus musculus', 'Danio rerio', 
            'Pan troglodytes', 'Canis lupus'
        ]
        self.protein_data = {}

    def fetch_protein_sequences(self):
        """
        Fetch protein sequences from UniProt using their REST API.
        """
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        taxonomy_ids = {
            "Homo sapiens": 9606,
            "Mus musculus": 10090,
            "Danio rerio": 7955,
            "Pan troglodytes": 9598,
            "Canis lupus": 9612
        }
        sequences = {}

        for species, tax_id in taxonomy_ids.items():
            query = f"({self.protein_name}) AND (taxonomy_id:{tax_id})"
            params = {
                "query": query,
                "format": "fasta",
                "size": 1  # Limit to 1 result per species
            }
            
            try:
                # Send request to UniProt API
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                if response.text.strip():
                    # Parse FASTA data
                    lines = response.text.strip().split("\n")
                    header = lines[0]
                    sequence = "".join(lines[1:])
                    
                    # Create SeqRecord
                    seq_record = SeqRecord(
                        Seq(sequence),
                        id=species.replace(" ", "_"),
                        description=header
                    )
                    sequences[species] = seq_record
                else:
                    print(f"No sequence found for {species}")
            except requests.exceptions.RequestException as e:
                print(f"Error fetching data for {species}: {e}")
        
        return sequences


    def create_multiple_alignment(self, sequences: Dict[str, SeqRecord]) -> str:
        """
        Create multiple sequence alignment with error handling
        """
        valid_sequences = {k: v for k, v in sequences.items() if v is not None and len(v.seq) > 0}
        
        if len(valid_sequences) < 2:
            raise ValueError("Not enough valid sequences for alignment")
        
        min_length = min(len(seq.seq) for seq in valid_sequences.values())
        truncated_sequences = [
            seq[:min_length] for seq in valid_sequences.values()
        ]
        
        alignment = Align.MultipleSeqAlignment(truncated_sequences)
        
        alignment_file = f"{self.protein_name}_protein_alignment.fasta"
        AlignIO.write(alignment, alignment_file, "fasta")
        
        return alignment_file

    def fetch_conservation_data(self, alignment_file: str):
        """
        Calculate and visualize conservation scores from protein alignment
        """
        try:
            # Read alignment
            alignment = AlignIO.read(alignment_file, "fasta")
            
            # Calculate conservation scores
            conservation_scores = self._calculate_protein_conservation(alignment)
            
            # Save conservation data
            conservation_df = pd.DataFrame(conservation_scores)
            conservation_df.to_csv(
                f"{self.protein_name}_protein_conservation.csv", 
                index=False
            )
            
            # Print conservation scores
            print("\nProtein Conservation Scores Summary:")
            print(conservation_df.describe())
            
            # Visualize conservation scores
            self._visualize_conservation_scores(conservation_scores)
            
            print("Protein conservation analysis completed successfully.")
            return conservation_scores
        
        except Exception as e:
            print(f"Protein conservation data retrieval error: {e}")
            return None

    def _calculate_protein_conservation(self, alignment):
        """
        Calculate conservation scores for protein sequences using BLOSUM62 matrix
        """
        conservation_scores = []
        blosum62 = MatrixInfo.blosum62
        
        for pos in range(alignment.get_alignment_length()):
            column = alignment[:, pos]
            
            # Count amino acid frequencies
            freq = {}
            for aa in column:
                freq[aa] = freq.get(aa, 0) + 1
            
            # Calculate conservation using BLOSUM62 score matrix
            similarity_scores = []
            for aa1 in set(column):
                aa_scores = [blosum62.get((aa1, aa2), blosum62.get((aa2, aa1), 0)) 
                              for aa2 in set(column)]
                similarity_scores.extend(aa_scores)
            
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
            conservation_score = (avg_similarity + 4) / 10  # Normalize to 0-1 range
            
            conservation_scores.append({
                'position': pos,
                'conservation_score': conservation_score,
                'most_common_amino_acid': max(freq, key=freq.get)
            })
        
        return conservation_scores

    def _visualize_conservation_scores(self, conservation_scores):
        """
        Create visualization of protein conservation scores
        """
        df = pd.DataFrame(conservation_scores)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df['position'], df['conservation_score'], linewidth=2)
        plt.title(f'Conservation Scores for {self.protein_name} Protein')
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
        plt.savefig(f'{self.protein_name}_protein_conservation_plot.png')
        plt.close()

def main():
    # Email is required for NCBI Entrez
    protein_miner = ProteinDataMiner('BRCA1', 'divy.moodi@gmail.com')
    
    try:
        # Fetch protein sequences
        sequences = protein_miner.fetch_protein_sequences()
        
        # Create multiple sequence alignment
        alignment_file = protein_miner.create_multiple_alignment(sequences)
        
        # Fetch and visualize conservation data
        conservation_scores = protein_miner.fetch_conservation_data(alignment_file)
        
        # Print detailed information about conservation
        if conservation_scores:
            print("\nDetailed Protein Conservation Scores:")
            for score in conservation_scores[:10]:  # Print first 10 scores
                print(f"Position {score['position']}: Score = {score['conservation_score']:.4f}, "
                      f"Most Common Amino Acid = {score['most_common_amino_acid']}")
        
    except Exception as e:
        print(f"An error occurred during protein analysis: {e}")

if __name__ == '__main__':
    main()