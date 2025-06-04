import unittest

import pandas as pd
import numpy as np
from helper_functions import (
    GffEntry,
    split_read,
    map_read_to_gene,
    generate_count_matrix,
    filter_matrix,
    normalize_expressions,
)


class TestSplitGene(unittest.TestCase):
    def test_split_gene(self):
        read = "AATTTTCACGGTTTTTTTTTTTTTTTTTTTTTTTTTACTTTGTAAGAATAATAATGAGGCTTTGGC"

        result = split_read(read)
        self.assertEqual(len(result), 2)
        barcode, seq = result
        self.assertEqual(barcode, "AATTTTCACGGT")
        self.assertEqual(seq, "ACTTTGTAAGAATAATAATGAGGCTTTGGC")


class TestMapReadToGene(unittest.TestCase):
    def test_map_read_to_gene_with_error(self):
        read = "BBBB"
        ref_seq = "AA-AXXXXBB-BXXXXCCCCC"
        genes = {
            "A": GffEntry(".", ".", ".", 0, 4, ".", ".", ".", {"gene_name": "A"}),
            "B": GffEntry(".", ".", ".", 8, 12, ".", ".", ".", {"gene_name": "B"}),
            "C": GffEntry(".", ".", ".", 16, 20, ".", ".", ".", {"gene_name": "C"}),
        }

        result = map_read_to_gene(read, ref_seq, genes)
        self.assertEqual(len(result), 2)
        gene, similarity = result
        self.assertEqual(gene, "B")
        self.assertEqual(similarity, 0.75)

    def test_map_read_to_gene_no_match(self):
        read = "XXXX"
        ref_seq = "AA-AXXXXBB-BXXXXCCCCC"
        genes = {
            "A": GffEntry(".", ".", ".", 0, 4, ".", ".", ".", {"gene_name": "A"}),
            "B": GffEntry(".", ".", ".", 8, 12, ".", ".", ".", {"gene_name": "B"}),
            "C": GffEntry(".", ".", ".", 16, 20, ".", ".", ".", {"gene_name": "C"}),
        }

        result = map_read_to_gene(read, ref_seq, genes)
        self.assertEqual(len(result), 2)
        gene, similarity = result
        self.assertEqual(gene, None)
        self.assertEqual(similarity, 1.0)


class TestGenerateCountMatrix(unittest.TestCase):
    def test_generate_count_matrix(self):
        reads = [
            "AAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTAAAA",
            "AAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTAAAA",
            "AAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTBBBB",
            "AAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTCCCC",
            "BBBBBBBBBBBBTTTTTTTTTTTTTTTTTTTTTTTTBBBB",
            "BBBBBBBBBBBBTTTTTTTTTTTTTTTTTTTTTTTTBBBB",
        ]
        ref_seq = "AAAAXXXXBBBBXXXXCCCC"
        genes = {
            "A": GffEntry(".", ".", ".", 0, 4, ".", ".", ".", {"gene_name": "A"}),
            "B": GffEntry(".", ".", ".", 8, 12, ".", ".", ".", {"gene_name": "B"}),
            "C": GffEntry(".", ".", ".", 16, 20, ".", ".", ".", {"gene_name": "C"}),
        }

        counts = generate_count_matrix(reads, ref_seq, genes, similarity_threshold=0.99)
        counts = counts.sort_index()
        columns = counts.columns.sort_values()
        counts = counts[columns]

        expected = pd.DataFrame(
            index=["AAAAAAAAAAAA", "BBBBBBBBBBBB"],
            columns=["A", "B", "C"],
            data=np.array([[2, 1, 1], [0, 2, 0]]),
        )

        np.testing.assert_almost_equal(counts.values, expected.values)
        self.assertTrue(all(expected.index == counts.index))
        self.assertTrue(all(expected.columns == counts.columns))


class TestFilterMatrix(unittest.TestCase):
    def test_filter_matrix(self):
        mtx = pd.DataFrame(
            index=[1, 2, 3, 4, 5, 6, 7],
            columns=["A", "B", "C", "D", "E"],
            data=np.array(
                [
                    [2, 1, 0, 1, 0],
                    [0, 2, 2, 0, 0],
                    [3, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 3, 0, 0, 0],
                    [5, 3, 0, 0, 0],
                ]
            ),
        )
        min_gene_count = 2
        min_cell_count = 2
        expected = pd.DataFrame(
            index=[1, 2, 3, 6, 7],
            columns=["A", "B", "C"],
            data=np.array(
                [
                    [2, 1, 0],
                    [0, 2, 2],
                    [3, 2, 0],
                    [1, 3, 0],
                    [5, 3, 0],
                ]
            ),
        )

        filtered_mtx = filter_matrix(
            mtx,
            min_counts_per_cell=min_cell_count,
            min_counts_per_gene=min_gene_count,
        )
        filtered_mtx = filtered_mtx.sort_index()
        columns = filtered_mtx.columns.sort_values()
        filtered_mtx = filtered_mtx[columns]

        np.testing.assert_almost_equal(filtered_mtx.values, expected.values)
        self.assertTrue(all(expected.index == filtered_mtx.index))
        self.assertTrue(all(expected.columns == filtered_mtx.columns))


class TestNormalizeExpressions(unittest.TestCase):
    def test_normalize_expressions(self):

        mtx = pd.DataFrame(
            index=np.array([1, 2, 3]),
            columns=np.array(["A", "B", "C", "D"]),
            data=np.array(
                [
                    [0, np.e - 1, np.e ** 2 - 1, np.e ** 7 - 1],
                    [1, 1, 1, 1],
                    [1, 0, 0, 0],
                ]
            ),
        )
        expected = pd.DataFrame(
            index=[1, 2, 3],
            columns=["A", "B", "C", "D"],
            data=np.array(
                [
                    [0, 1000, 2000, 7000],
                    [2500, 2500, 2500, 2500],
                    [10000, 0, 0, 0],
                ]
            ),
        )

        norm_mtx = normalize_expressions(mtx)

        np.testing.assert_equal(norm_mtx.values, expected)


if __name__ == "__main__":
    unittest.main()
