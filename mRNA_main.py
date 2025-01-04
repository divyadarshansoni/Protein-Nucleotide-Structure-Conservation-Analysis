import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import Entrez, AlignIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import Dict, List

class GenomicDataMiner:
    def __init__(self, gene_name: str, email: str):
        Entrez.email = email
        self.gene_name = gene_name
        self.species_list = [
            'Homo sapiens', 'Mus musculus', 'Danio rerio', 
            'Pan troglodytes', 'Canis lupus'
        ]
        self.gene_data = {}

    def fetch_gene_sequences(self) -> Dict[str, SeqRecord]:
        """
        Fetch gene sequences with multiple fallback strategies
        """
        sequences = {}
        
        for species in self.species_list:
            try:
                search_terms = [
                    f"{species}[Organism] AND {self.gene_name}[Gene] AND RefSeq[Filter]",
                    f"{species}[Organism] AND {self.gene_name}",
                    f"{self.gene_name} gene"
                ]
                
                for term in search_terms:
                    try:
                        search_handle = Entrez.esearch(
                            db="nucleotide", 
                            term=term,
                            retmax=1
                        )
                        search_record = Entrez.read(search_handle)
                        
                        if search_record["IdList"]:
                            fetch_handle = Entrez.efetch(
                                db="nucleotide", 
                                id=search_record["IdList"][0], 
                                rettype="fasta", 
                                retmode="text"
                            )
                            sequence_record = fetch_handle.read().strip().split('\n')
                            header = sequence_record[0]
                            sequence = ''.join(sequence_record[1:]).replace(' ', '')
                            
                            seq_record = SeqRecord(
                                Seq(sequence), 
                                id=species.replace(' ', '_'), 
                                description=header
                            )
                            sequences[species] = seq_record
                            fetch_handle.close()
                            break
                        
                    except Exception as inner_e:
                        print(f"Inner search failed for {species}: {inner_e}")
                
                search_handle.close()
            except Exception as e:
                print(f"Error fetching sequence for {species}: {e}")
        
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
        
        alignment_file = f"{self.gene_name}_alignment.fasta"
        AlignIO.write(alignment, alignment_file, "fasta")
        
        return alignment_file

    def fetch_conservation_data(self, alignment_file: str):
        """
        Calculate and visualize conservation scores from alignment
        """
        try:
            # Read alignment
            alignment = AlignIO.read(alignment_file, "fasta")
            
            # Calculate conservation scores
            conservation_scores = self._calculate_conservation(alignment)
            
            # Save conservation data
            conservation_df = pd.DataFrame(conservation_scores)
            conservation_df.to_csv(
                f"{self.gene_name}_conservation.csv", 
                index=False
            )
            
            # Print conservation scores
            print("\nConservation Scores Summary:")
            print(conservation_df.describe())
            
            # Visualize conservation scores
            self._visualize_conservation_scores(conservation_scores)
            
            print("Conservation analysis completed successfully.")
            return conservation_scores
        
        except Exception as e:
            print(f"Conservation data retrieval error: {e}")
            return None

    def _calculate_conservation(self, alignment):
        """
        Calculate conservation scores using Shannon entropy
        """
        conservation_scores = []
        for pos in range(alignment.get_alignment_length()):
            column = alignment[:, pos]
            
            # Count nucleotide frequencies
            freq = {}
            for nt in column:
                freq[nt] = freq.get(nt, 0) + 1
            
            # Avoid division by zero
            unique_chars = len(set(column))
            if unique_chars <= 1:
                conservation_score = 1.0
            else:
                # Calculate conservation score (Shannon entropy-based)
                total = len(column)
                entropy = sum(
                    -(count/total) * (np.log2(count/total) if count > 0 else 0) 
                    for count in freq.values()
                )
                conservation_score = 1 - (entropy / np.log2(unique_chars))
            
            conservation_scores.append({
                'position': pos,
                'conservation_score': conservation_score,
                'most_common_nucleotide': max(freq, key=freq.get)
            })
        
        return conservation_scores

    def _visualize_conservation_scores(self, conservation_scores):
        """
        Create visualization of conservation scores
        """
        df = pd.DataFrame(conservation_scores)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df['position'], df['conservation_score'], linewidth=2)
        plt.title(f'Conservation Scores for {self.gene_name} Gene')
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
        plt.savefig(f'{self.gene_name}_conservation_plot.png')
        plt.close()

def main():
    # Email is required for NCBI Entrez
    gene_miner = GenomicDataMiner('BRCA1', 'divy.moodi@gmail.com')
    
    try:
        # Fetch gene sequences
        sequences = gene_miner.fetch_gene_sequences()
        
        # Create multiple sequence alignment
        alignment_file = gene_miner.create_multiple_alignment(sequences)
        
        # Fetch and visualize conservation data
        conservation_scores = gene_miner.fetch_conservation_data(alignment_file)
        
        # Print detailed information about conservation
        if conservation_scores:
            print("\nDetailed Conservation Scores:")
            for score in conservation_scores[:10]:  # Print first 10 scores
                print(f"Position {score['position']}: Score = {score['conservation_score']:.4f}, "
                      f"Most Common Nucleotide = {score['most_common_nucleotide']}")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == '__main__':
    main()