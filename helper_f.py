from collections import namedtuple
from typing import Dict, List, Tuple
import pandas as pd
import math
from Bio import Align


GffEntry = namedtuple(
    "GffEntry",
    [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ],
)


GeneDict = Dict[str, GffEntry]


def read_gff(fname: str) -> Dict[str, GffEntry]:
    gene_dict = {}

    with open(fname) as f:
        for line in f:
            if line.startswith("#"):  # Comments start with '#' character
                continue

            parts = line.split("\t")
            parts = [p.strip() for p in parts]

            # Convert start and stop to ints
            start_idx = GffEntry._fields.index("start")
            parts[start_idx] = int(parts[start_idx]) - 1  # GFFs count from 1..
            stop_idx = GffEntry._fields.index("end")
            parts[stop_idx] = int(parts[stop_idx]) - 1  # GFFs count from 1..

            # Split the attributes
            attr_index = GffEntry._fields.index("attribute")
            attributes = {}
            for attr in parts[attr_index].split(";"):
                attr = attr.strip()
                k, v = attr.split("=")
                attributes[k] = v
            parts[attr_index] = attributes

            entry = GffEntry(*parts)

            gene_dict[entry.attribute["gene_name"]] = entry

    return gene_dict


def split_read(read: str) -> Tuple[str, str]:
    """Split a given read into its barcode and DNA sequence. The reads are
    already in DNA format, so no additional work will have to be done. This
    function needs only to take the read, and split it into the cell barcode,
    the primer, and the DNA sequence. The primer is not important, so we discard
    that.

    The first 12 bases correspond to the cell barcode.
    The next 24 bases corresond to the oligo-dT primer. (discard this)
    The reamining bases corresond to the actual DNA of interest.

    Parameters
    ----------
    read: str

    Returns
    -------
    str: cell_barcode
    str: mRNA sequence

    """
    cell_barcode = read[0:12]
    mRNA_sequence = read[36:]
    return (cell_barcode,mRNA_sequence)


def map_read_to_gene(read: str, ref_seq: str, genes: GeneDict) -> Tuple[str, float]:
    """Map a given read to a gene with a confidence score using Hamming distance.

    Parameters
    ----------
    read: str
        The DNA sequence to be aligned to the reference sequence. This should
        NOT include the cell barcode or the oligo-dT primer.
    ref_seq: str
        The reference sequence that the read should be aligned against.
    genes: GeneDict

    Returns
    -------
    gene: str
        The name of the gene (using the keys of the `genes` parameter, which the
        read maps to best. If the best alignment maps to a region that is not a
        gene, the function should return `None`.
    similarity: float
        The similarity of the aligned read. This is computed by taking the
        Hamming distance between the aligned read and the reference sequence.
        E.g. catac and cat-x will have similarity 3/5=0.6.


    """
    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.gap_score = -1
    aligner.extend_gap_score = -1

    aligned_sequences = aligner.align(ref_seq,read)
    max_similarity = 0
    best_start = None
    best_end = None

    if len(aligned_sequences) <= 10000:
        for elt in aligned_sequences:
            start = elt.aligned[0][0][0]    
            end = elt.aligned[0][::-1][0][1] 
            similarity = elt.score / (end - start)
            if similarity > max_similarity:
                max_similarity = similarity
                best_start = start
                best_end = end
    else:
        best_end = aligned_sequences[0].aligned[0][::-1][0][1]  
        best_start = aligned_sequences[0].aligned[0][0][0]  
        max_similarity = aligned_sequences[0].score / (best_end - best_start)
    read_gene = None

    # Find gene
    if best_start != None:    
        for key in genes.keys():
                gene = genes[key]
                lower_bound = gene.start
                upper_bound = gene.end
                if best_start >= lower_bound and best_end <= upper_bound:
                    read_gene = key
                    break

    return (read_gene, max_similarity)


def generate_count_matrix(
    reads: List[str], ref_seq: str, genes: GeneDict, similarity_threshold: float
) -> pd.DataFrame:
    """

    Parameters
    ----------
    reads: List[str]
        The list of all reads that will be aligned.
    ref_seq: str
        The reference sequence that the read should be aligned against.
    genes: GeneDict
    similarity_threshold: float

    Returns
    -------
    count_table: pd.DataFrame
        The count table should be an N x G matrix where N is the number of
        unique cell barcodes in the reads and G is the number of genes in
        `genes`. The dataframe columns should be to a list of strings
        corrsponding to genes and the dataframe index should be a list of
        strings corresponding to cell barcodes. Each cell in the matrix should
        indicate the number of times a read mapped to a gene in that particular
        cell.

    """
    read_dict = {}
    counter = 0 
    for read in reads:
        barcode = str(split_read(read)[0])
        read_dict[barcode] = []

    for read in reads:
        [barcode, mrna_seq] =  split_read(read)
        read_to_gene = map_read_to_gene(mrna_seq, ref_seq, genes)
        if read_to_gene[1] >= similarity_threshold:
                read_dict[str(barcode)].append(read_to_gene)
        counter += 1

    index = list(read_dict.keys())
    columns = list(genes.keys())

    data = [[0 for i in range(len(columns))] for j in range(len(index))]

    for i in range(len(index)):
        mapped_genes = read_dict[index[i]]
        for mapped_gene in mapped_genes:
            if mapped_gene[0] != None:
                gene_index = columns.index(mapped_gene[0])
                data[i][gene_index] += 1 

    return pd.DataFrame(index = index,columns = columns,data = data,)


def filter_matrix(
    count_matrix: pd.DataFrame,
    min_counts_per_cell: float,
    min_counts_per_gene: float,
) -> pd.DataFrame:
    """Filter a matrix by cell counts and gene counts.
    The cell count is the total number of molecules sequenced for a particular
    cell. The gene count is the total number of molecules sequenced that
    correspond to a particular gene. Filtering statistics should be computed on
    the original matrix. E.g. if you filter out the genes first, the filtered
    gene molecules should still count towards the cell counts.

    Parameters
    ----------
    count_matrix: pd.DataFrame
    min_counts_per_cell: float
    min_counts_per_gene: float

    Returns
    -------
    filtered_count_matrix: pd.DataFrame

    """
    original_indices = list(count_matrix.index)
    original_columns = list(count_matrix.columns)
    original_matrix = count_matrix.values
    keep_indices= []
    for i in range(len(original_matrix)):
        if sum(original_matrix[i]) >= min_counts_per_cell:
            keep_indices.append(i)

    transposed_matrix = original_matrix.transpose()
    keep_columns = []
    for i in range(len(transposed_matrix)):
        if sum(transposed_matrix[i]) >= min_counts_per_gene:
            keep_columns.append(i)

    # Get new indexes
    new_indices = []
    for i in keep_indices:
        new_indices.append(original_indices[i])
    # Get new columns     
    new_columns = []
    for i in keep_columns:
        new_columns.append(original_columns[i])

    new_matrix = []
    for i in keep_indices:
        row = original_matrix[i]
        new_row = []
        for j in keep_columns:
            new_row.append(row[j])
        new_matrix.append(new_row)

    return pd.DataFrame(index = new_indices, columns = new_columns, data = new_matrix,)


def normalize_expressions(expression_data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expressions by applying natural log-transformation with pseudo count 1,
    and scaling expressions of each sample to sum up to 10000.

    Parameters
    ----------
    expression_data: pd.DataFrame
        Expression matrix with cells as rows and genes as columns.

    Returns
    -------
    normalized_data: pd.DataFrame
        Normalized expression matrix with cells as rows and genes as columns.
        Matrix should have the same shape as the input matrix.
        Matrix should have the same index and column labels as the input matrix.
        Order of rows and columns should remain the same.
        Values in the matrix should be positive or zero.
    """
    original_indices = list(expression_data.index)
    original_columns = list(expression_data.columns)
    original_matrix = expression_data.values


    for i in range(len(original_matrix)):
        cell = original_matrix[i]
        for j in range(len(cell)):
            original_matrix[i][j] = math.log(original_matrix[i][j] + 1)
        vsota = sum(original_matrix[i])
        for j in range(len(cell)):
            original_matrix[i][j] /= vsota
        for j in range(len(cell)):
            original_matrix[i][j] *= 10000
    return pd.DataFrame(index = original_indices, columns = original_columns, data = original_matrix,)


def hypergeometric_pval(N: int, n: int, K: int, k: int) -> float:
    """
    Calculate the p-value using the following hypergeometric distribution.

    Parameters
    ----------
    N: int
        Total number of genes in the study (gene expression matrix)
    n: int
        Number of genes in your proposed gene set (e.g. from differential expression)
    K: int
        Number of genes in an annotated gene set (e.g. GO gene set)
    k: int
        Number of genes in both annotated and proposed geneset

    Returns
    -------
    p_value: float
        p-value from hypergeometric distribution of finding such or
        more extreme match at random
    """
    return (math.comb(K, k) * math.comb(N - K, n - k)) / (N, n)
