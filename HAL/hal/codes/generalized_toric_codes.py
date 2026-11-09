import copy
import functools
import itertools
import math
from collections.abc import Collection, Sequence
from math import gcd
from typing import Tuple

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import sympy
from qldpc.codes.common import ClassicalCode, CSSCode
from qldpc.objects import Node, abstract
from scipy.sparse import csr_matrix, lil_matrix


class GeneralizedToricCode(CSSCode):
    """Generalized toric codes on twisted tori for quantum error correction."""

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | None = None,
        lattice_vectors: tuple[tuple[int, int], tuple[int, int]] | None = None,
    ) -> None:
        """Construct a generalized toric code on twisted torus."""

        # Set up field first
        if field is None:
            field = 2

        # Ensure both x and y symbols are present in the polynomials
        x, y = sympy.symbols("x y")
        poly_a_expanded = poly_a + 0 * x + 0 * y
        poly_b_expanded = poly_b + 0 * x + 0 * y

        # Convert dictionary orders to sequence if needed
        if isinstance(orders, dict):
            if x not in orders or y not in orders:
                raise ValueError("Orders dictionary must contain both x and y symbols")
            orders_seq = [orders[x], orders[y]]
        else:
            orders_seq = list(orders)
            if len(orders_seq) != 2:
                raise ValueError(f"Expected exactly 2 orders, got {len(orders_seq)}")

        # Set up lattice vectors for twisted torus
        if lattice_vectors is None:
            # Default to rectangular torus
            beta, alpha = orders_seq  # x order, y order
            lattice_vectors = ((0, alpha), (beta, 0))
        else:
            # Verify lattice vectors are compatible with orders
            a1, a2 = lattice_vectors
            if a1[0] != 0:
                raise ValueError("First lattice vector must be of form (0, α)")

            alpha, beta, gamma = a1[1], a2[0], a2[1]

        # Store lattice information before parent initialization
        self.lattice_vectors = lattice_vectors

        # Try custom matrix construction first
        # Build parity check matrices with correct twisted geometry
        matrix_x, matrix_z = self._build_custom_matrices(
            orders_seq, poly_a_expanded, poly_b_expanded, field
        )

        self.Hx = matrix_x
        self.Hz = matrix_z

        # Remove redundant stabilizers if requested
        # if auto_remove_redundant:
        #     matrix_x, matrix_z = self._remove_redundant_from_matrices(matrix_x, matrix_z, field)

        # Convert galois arrays to regular numpy arrays for CSSCode
        if hasattr(matrix_x, "view"):
            matrix_x = matrix_x.view(np.ndarray)
        if hasattr(matrix_z, "view"):
            matrix_z = matrix_z.view(np.ndarray)

        # Initialize parent CSSCode with the matrices
        CSSCode.__init__(
            self,
            code_x=matrix_x,
            code_z=matrix_z,
            field=field,
            promise_equal_distance_xz=True,
        )

        # Store additional information after parent initialization
        self.symbols = (x, y)
        self.orders = tuple(orders_seq)
        self.poly_a = sympy.Poly(poly_a_expanded)
        self.poly_b = sympy.Poly(poly_b_expanded)

    def _build_custom_matrices(
        self,
        orders_seq: Sequence[int],
        poly_a: sympy.Basic | sympy.Poly,
        poly_b: sympy.Basic | sympy.Poly,
        field: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build custom parity check matrices for twisted torus."""
        # Get lattice parameters
        alpha = self.lattice_vectors[0][1]  # from a1 = (0, α)
        beta = self.lattice_vectors[1][0]  # from a2 = (β, γ)
        gamma = self.lattice_vectors[1][1]  # from a2 = (β, γ)

        num_qubits = 2 * alpha * beta
        num_checks = alpha * beta

        # Set up Galois field
        GF = galois.GF(field)

        # Extract polynomial terms
        terms_a = self._extract_terms(poly_a, field)
        terms_b = self._extract_terms(poly_b, field)

        # Build matrices
        Hx = GF.Zeros((num_checks, num_qubits))
        Hz = GF.Zeros((num_checks, num_qubits))

        # For each stabilizer position
        self._graph = nx.Graph()
        for stabilizer_idx in range(num_checks):
            row = stabilizer_idx // beta
            col = stabilizer_idx % beta

            # X-stabilizer
            ind = int(2 * row * beta + col)
            node_c = Node(index=ind, is_data=False)
            for x_exp, y_exp, coeff in terms_a:
                qubit_idx = self._get_qubit_index(row, col, x_exp, y_exp, 0, alpha, beta, gamma)
                # Convert coeff to GF element
                coeff_gf = GF(coeff)
                Hx[stabilizer_idx, qubit_idx] = Hx[stabilizer_idx, qubit_idx] + coeff_gf

                node_d = Node(index=int(qubit_idx), is_data=True)
                self._graph.add_edge(node_c, node_d, val=Hx[stabilizer_idx, qubit_idx])
                # Hx[stabilizer_idx, qubit_idx] = Hx[stabilizer_idx, qubit_idx] + coeff_gf

            for x_exp, y_exp, coeff in terms_b:
                qubit_idx = self._get_qubit_index(row, col, x_exp, y_exp, 1, alpha, beta, gamma)
                # Convert coeff to GF element
                coeff_gf = GF(coeff)
                Hx[stabilizer_idx, qubit_idx] = Hx[stabilizer_idx, qubit_idx] + coeff_gf

                node_d = Node(index=int(qubit_idx), is_data=True)
                self._graph.add_edge(node_c, node_d, val=Hx[stabilizer_idx, qubit_idx])

            # Z-stabilizer
            ind = self._get_qubit_index(row, col, -1, 0, 1, alpha, beta, gamma)
            node_c = Node(index=ind, is_data=False)
            for x_exp, y_exp, coeff in terms_b:
                qubit_idx = self._get_qubit_index(row, col, -x_exp, -y_exp, 0, alpha, beta, gamma)
                # Convert coeff to GF element
                # neg_coeff = (-coeff) % field
                # neg_coeff_gf = GF(neg_coeff)
                neg_coeff_gf = GF((-coeff) % field)
                Hz[stabilizer_idx, qubit_idx] = Hz[stabilizer_idx, qubit_idx] + neg_coeff_gf

                node_d = Node(index=int(qubit_idx), is_data=True)
                self._graph.add_edge(node_c, node_d, val=Hz[stabilizer_idx, qubit_idx])

            for x_exp, y_exp, coeff in terms_a:
                qubit_idx = self._get_qubit_index(row, col, -x_exp, -y_exp, 1, alpha, beta, gamma)
                # Compute -coeff modulo field first, then convert to GF element
                neg_coeff = (-coeff) % field
                neg_coeff_gf = GF(neg_coeff)
                Hz[stabilizer_idx, qubit_idx] = Hz[stabilizer_idx, qubit_idx] + neg_coeff_gf

                node_d = Node(index=int(qubit_idx), is_data=True)
                self._graph.add_edge(node_c, node_d, val=Hz[stabilizer_idx, qubit_idx])

        return Hx, Hz

    def _extract_terms(
        self, poly: sympy.Basic | sympy.Poly, field: int
    ) -> list[tuple[int, int, int]]:
        """Extract polynomial terms from Laurent polynomial."""
        terms = []

        # Convert to sympy expression first
        if isinstance(poly, sympy.Poly):
            expr = poly.as_expr()
        else:
            expr = poly

        # Expand the expression
        expr = sympy.expand(expr)

        # Get the symbols
        x, y = sympy.symbols("x y")

        # Handle Laurent polynomials manually
        if expr.is_Add:
            # Multiple terms
            addends = expr.args
        else:
            # Single term
            addends = [expr]

        for term in addends:
            # Extract coefficient and powers
            coeff, x_exp, y_exp = self._extract_term_info(term, x, y)

            if coeff != 0:
                terms.append((int(x_exp), int(y_exp), int(coeff) % field))

        return terms

    def _extract_term_info(
        self, term: sympy.Basic, x: sympy.Symbol, y: sympy.Symbol
    ) -> tuple[int, int, int]:
        """Extract coefficient and exponents from a single term."""
        # Initialize
        coeff = 1
        x_exp = 0
        y_exp = 0

        # Handle different term types
        if term.is_Number:
            # Constant term
            coeff = int(term)
        elif term == x:
            # Just x
            coeff = 1
            x_exp = 1
        elif term == y:
            # Just y
            coeff = 1
            y_exp = 1
        elif term.is_Mul:
            # Product of factors
            for factor in term.args:
                if factor.is_Number:
                    coeff *= int(factor)
                elif factor == x:
                    x_exp += 1
                elif factor == y:
                    y_exp += 1
                elif factor.is_Pow:
                    base, exp = factor.args
                    if base == x:
                        x_exp += int(exp)
                    elif base == y:
                        y_exp += int(exp)
                elif factor.has(x) and factor.is_Pow:
                    # Handle x^(-1) type terms
                    base, exp = factor.args
                    if base == x:
                        x_exp += int(exp)
                elif factor.has(y) and factor.is_Pow:
                    # Handle y^(-1) type terms
                    base, exp = factor.args
                    if base == y:
                        y_exp += int(exp)
        elif term.is_Pow:
            # Single power term
            base, exp = term.args
            if base == x:
                x_exp = int(exp)
            elif base == y:
                y_exp = int(exp)
            coeff = 1
        elif term == x:
            x_exp = 1
            coeff = 1
        elif term == y:
            y_exp = 1
            coeff = 1

        return coeff, x_exp, y_exp

    def _qubit_index_twisted(self, row: int, col: int, edge_type: int, alpha, beta, gamma) -> int:
        """Get qubit index for edge at (row, col) on twisted torus."""
        n = 2 * alpha * beta
        base_index = row * (2 * beta) + col

        old_block = base_index // beta
        new_block = (base_index - 1) // beta
        boundary_correction = (2 * gamma + 1) * beta * (new_block - old_block)

        if edge_type == 0:
            return (base_index - 1 - boundary_correction) % n  # horizontal edge
        return (base_index - beta) % (n)

    def y_shift(self, qubit_index: int, shift_y: int, alpha: int, beta: int) -> int:
        """Apply y-direction shift using Eq. (B1)"""
        n = 2 * alpha * beta
        return (qubit_index + (2 * beta) * shift_y) % n

    def x_shift(self, qubit_index: int, shift_x: int, alpha: int, beta: int, gamma: int) -> int:
        """Apply x-direction shift using Eq. (B4)"""
        n = 2 * alpha * beta
        if shift_x == 0:
            return qubit_index

        old_block = qubit_index // beta
        new_block = (qubit_index + shift_x) // beta
        boundary_correction = (2 * gamma + 1) * beta * (new_block - old_block)

        result = (qubit_index + shift_x - boundary_correction) % n
        return result

    def _get_qubit_index(
        self,
        base_row: int,
        base_col: int,
        shift_x: int,
        shift_y: int,
        edge_type: int,
        alpha: int,
        beta: int,
        gamma: int,
    ) -> int:
        """Get qubit index with twisted boundary conditions."""
        qubit_index = self._qubit_index_twisted(
            base_row, base_col, edge_type, alpha, beta, gamma
        )  # Start with horizontal edge
        y_shifted_qubit_index = self.y_shift(qubit_index, shift_y, alpha, beta)
        shifted_qubit = self.x_shift(y_shifted_qubit_index, shift_x, alpha, beta, gamma)
        return shifted_qubit

    def _remove_redundant_from_matrices(self, Hx: np.ndarray, Hz: np.ndarray, field: int):
        """Remove redundant stabilizers."""
        # Convert to galois arrays if needed
        GF = galois.GF(field)
        if not isinstance(Hx, galois.Array):
            Hx = GF(Hx)
        if not isinstance(Hz, galois.Array):
            Hz = GF(Hz)

        # Find independent rows using galois field operations
        independent_x = self._find_independent_rows(Hx)
        independent_z = self._find_independent_rows(Hz)

        # Extract independent rows
        Hx_reduced = Hx[independent_x, :]
        Hz_reduced = Hz[independent_z, :]

        removed_x = len(Hx) - len(independent_x)
        removed_z = len(Hz) - len(independent_z)

        # if removed_x > 0 or removed_z > 0:
        #     print(f"Removed {removed_x} redundant X-stabilizers and {removed_z} redundant Z-stabilizers")

        return Hx_reduced, Hz_reduced

    def _find_independent_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Find linearly independent rows using row reduction."""
        if matrix.size == 0:
            return np.array([], dtype=int)

        # Use galois field row reduction
        m, n = matrix.shape
        working_matrix = matrix.copy()
        independent_rows = []

        current_row = 0
        for col in range(n):
            if current_row >= m:
                break

            # Find pivot
            pivot_row = None
            for row in range(current_row, m):
                if working_matrix[row, col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            # Swap rows if needed
            if pivot_row != current_row:
                working_matrix[[current_row, pivot_row]] = working_matrix[[pivot_row, current_row]]

            independent_rows.append(current_row)

            # Eliminate
            pivot_val = working_matrix[current_row, col]
            for row in range(m):
                if row != current_row and working_matrix[row, col] != 0:
                    factor = working_matrix[row, col] / pivot_val
                    working_matrix[row] = working_matrix[row] - factor * working_matrix[current_row]

            current_row += 1

        return np.array(independent_rows)

    # Properties for accessing lattice parameters
    @property
    def is_twisted(self) -> bool:
        """Check if this is a twisted torus (γ ≠ 0)."""
        return self.lattice_vectors[1][1] != 0

    @property
    def alpha(self) -> int:
        """Get α parameter from lattice vector a1 = (0, α)."""
        return self.lattice_vectors[0][1]

    @property
    def beta(self) -> int:
        """Get β parameter from lattice vector a2 = (β, γ)."""
        return self.lattice_vectors[1][0]

    @property
    def gamma(self) -> int:
        """Get γ parameter from lattice vector a2 = (β, γ)."""
        return self.lattice_vectors[1][1]

    @property
    def graph(self) -> nx.Graph:
        """Get the graph representation of this code."""
        return self._graph

    def num_redundant_stabilizers(self) -> tuple[int, int]:
        """Get the number of redundant X and Z stabilizers."""
        try:
            Hx = self.matrix_x
            Hz = self.matrix_z

            # Handle both galois arrays and regular numpy arrays
            if hasattr(Hx, "view") and hasattr(Hx, "dtype"):
                # Galois array
                rank_x = np.linalg.matrix_rank(Hx.view(np.ndarray).astype(float))
                rank_z = np.linalg.matrix_rank(Hz.view(np.ndarray).astype(float))
            else:
                # Regular array
                rank_x = np.linalg.matrix_rank(Hx.astype(float))
                rank_z = np.linalg.matrix_rank(Hz.astype(float))

            redundant_x = Hx.shape[0] - rank_x
            redundant_z = Hz.shape[0] - rank_z

            return int(redundant_x), int(redundant_z)
        except Exception as e:
            print(f"Error calculating redundant stabilizers: {e}")
            return 0, 0

    def has_redundant_stabilizers(self) -> bool:
        """Check if the code has any redundant stabilizers."""
        redundant_x, redundant_z = self.num_redundant_stabilizers()
        return redundant_x > 0 or redundant_z > 0

    def get_stabilizer_ranks(self) -> tuple[int, int]:
        """Get the ranks of the X and Z stabilizer matrices."""
        Hx = self.matrix_x
        Hz = self.matrix_z

        if hasattr(Hx, "view"):
            rank_x = np.linalg.matrix_rank(Hx.view(np.ndarray).astype(float))
            rank_z = np.linalg.matrix_rank(Hz.view(np.ndarray).astype(float))
        else:
            rank_x = np.linalg.matrix_rank(Hx.astype(float))
            rank_z = np.linalg.matrix_rank(Hz.astype(float))

        return int(rank_x), int(rank_z)

    def get_qubit_pos(self, qubit: Node | tuple[str, int, int]) -> tuple[int, int]:
        """Get the canonical position of a qubit in this code."""
        if qubit.is_data:
            if (qubit.index // self.beta) % 2 == 0:
                return ((qubit.index // self.beta) % 2) + (
                    qubit.index % self.beta
                ) * 2 + 1, qubit.index // (self.beta)

            return ((qubit.index // self.beta) % 2) + (
                qubit.index % self.beta
            ) * 2 - 1, qubit.index // (self.beta)

        return ((qubit.index // self.beta) % 2) + (qubit.index % self.beta) * 2, qubit.index // (
            self.beta
        )

    def __str__(self) -> str:
        """Human-readable representation of this code."""
        text = ""
        if hasattr(self, "field") and self.field.order == 2:
            text += f"GeneralizedToricCode on {self.num_qubits} qubits"
        else:
            text += f"GeneralizedToricCode on {self.num_qudits} qudits over {self.field_name}"

        # Include lattice vector information
        a1, a2 = self.lattice_vectors
        text += f" with lattice vectors a1={a1}, a2={a2}"
        if self.is_twisted:
            text += f" (twisted torus with γ={self.gamma})"

        if hasattr(self, "symbols") and hasattr(self, "orders"):
            orders = dict(zip(self.symbols, self.orders))
            text += f", cyclic group orders {orders}, and generating polynomials"
            text += f"\n  A = {self.poly_a.as_expr()}"
            text += f"\n  B = {self.poly_b.as_expr()}"

        return text

    @classmethod
    def from_paper_parameters(
        cls,
        a: int,
        b: int,
        c: int,
        d: int,
        lattice_vectors: tuple[tuple[int, int], tuple[int, int]],
        field: int | None = None,
    ) -> "GeneralizedToricCode":
        """Construct GeneralizedToricCode from paper's (a,b,c,d)-generalized toric code parameters.

        Creates polynomials:
        f(x,y) = 1 + x + x^a * y^b
        g(x,y) = 1 + y + x^c * y^d

        Args:
            a, b, c, d: Exponents for generalized toric code
            lattice_vectors: ((0, α), (β, γ)) defining the twisted torus
            field: Base field
        """
        x, y = sympy.symbols("x y")

        # Build polynomials from paper's notation
        poly_f = 1 + x + x**a * y**b
        poly_g = 1 + y + x**c * y**d

        # Extract orders from lattice vectors
        alpha = lattice_vectors[0][1]  # from a1 = (0, α)
        beta = lattice_vectors[1][0]  # from a2 = (β, γ)

        # Use sequence instead of dictionary to avoid symbol ordering issues
        # The parent class expects symbols in alphabetical order: x, y
        orders = [beta, alpha]  # x gets beta, y gets alpha

        return cls(orders, poly_f, poly_g, field, lattice_vectors)
