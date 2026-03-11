#define _USE_MATH_DEFINES
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Numerical and qualitative checks for quantum graph theory bridges
// 1) RG flow exponents (z0, z1, alpha_eff, beta_eff)
// 2) Regge action on 2D triangular lattice (deficit angles)
// 3) Discrete U(1) curvature on square lattice

// ========== Parameters ==========
#define N 8         // Grid size for RG and EM lattice
#define TRI_N 10    // Size for triangular mesh
#define MAX_NODES 64
#define MAX_N 8
#define TRI_N_DOUBLE TRI_N*TRI_N
#define TRIS (TRI_N - 1) * (TRI_N - 1) * 2

// ================= Graph utilities ================
// Adjacency for 2D grid
int** build_grid_adj(int n) {
	int nodes = n * n;
	int** adj = (int**)malloc(nodes * sizeof(int*));
	for (int i = 0; i < nodes; ++i) {
		adj[i] = (int*)calloc(nodes, sizeof(int));
	}
	for (int r = 0; r < n; ++r) {
		for (int c = 0; c < n; ++c) {
			int u = r * n + c;
			if (r < n - 1) { int v = (r + 1) * n + c; adj[u][v] = adj[v][u] = 1; }
			if (c < n - 1) { int v = r * n + (c + 1); adj[u][v] = adj[v][u] = 1; }
		}
	}
	return adj;
}

// ========== RG flow exponents =============
double compute_avg_degree(int** adj, int nodes) {
	double sum = 0;
	for (int i = 0; i < nodes; ++i) {
		int deg = 0;
		for (int j = 0; j < nodes; ++j) deg += adj[i][j];
		sum += deg;
	}
	return sum / nodes;
}

void rg_flow() {
	int nodes = N * N;
	int** adj = build_grid_adj(N);
	double z0 = compute_avg_degree(adj, nodes);
	// Simple clustering: 4 blocks of size N/2 x N/2
	int clusters[MAX_NODES];
	for (int r = 0; r < N; ++r) for (int c = 0; c < N; ++c) {
		int idx = r*N + c;
		int cid = (r / (N / 2)) * 2 + (c / (N / 2));
		clusters[idx] = cid;
	}
	int sup_nodes = 4;
	int** sup_adj = (int**)malloc(sup_nodes * sizeof(int*));
	for (int i = 0; i < sup_nodes; ++i) sup_adj[i] =(int*) calloc(sup_nodes, sizeof(int));
	for (int u = 0; u < nodes; ++u) for (int v = u + 1; v < nodes; ++v) {
		if (adj[u][v] && clusters[u] != clusters[v]) {
			sup_adj[clusters[u]][clusters[v]]++;
			sup_adj[clusters[v]][clusters[u]]++;
		}
	}
	double z1 = compute_avg_degree(sup_adj, sup_nodes);
	double beta_eff = -log(z1 / z0) / log(2);
	double f_J = 0; int count = 0;
	for (int i = 0; i < sup_nodes; i++) for (int j = i + 1; j < sup_nodes; j++) if (sup_adj[i][j] > 0) { f_J += sup_adj[i][j]; count++; }
	f_J /= count;
	double alpha_eff = -log(f_J) / log(2);
	printf("RG flow: z0=%.3f, z1=%.3f, beta_eff=%.3f, f_J=%.3f, alpha_eff=%.3f\n", z0, z1, beta_eff, f_J, alpha_eff);
	// free memory...
}

// =========== Regge action on triangular mesh =============

double compute_triangle_area(double ax, double ay, double bx, double by, double cx, double cy) {
	return 0.5 * fabs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
}

double compute_deficit(double points[][2], int tris[][3], int n_tris, int v) {
	// Only interior v with 6 triangles assumed
	double angle_sum = 0;
	for (int t = 0; t < n_tris; ++t) {
		int i = tris[t][0], j = tris[t][1], k = tris[t][2];
		// check if v in tri
		if (i != v && j != v && k != v) continue;
		int a = i == v ? j : i;
		int b = v;
		int c = k == v ? j : k;
		double* A = points[a], * B = points[b], * C = points[c];
		double BAx = A[0] - B[0], BAy = A[1] - B[1];
		double BCx = C[0] - B[0], BCy = C[1] - B[1];
		double dot = BAx * BCx + BAy * BCy;
		double mag1 = hypot(BAx, BAy), mag2 = hypot(BCx, BCy);
		double cos_a = dot / (mag1 * mag2);
		angle_sum += acos(cos_a);
	}
	return 2 * M_PI - angle_sum;
}

void regge_triangular() {
	int size = TRI_N;
	int tot_pts = TRI_N_DOUBLE;
	double points[TRI_N_DOUBLE][2];
	int idx = 0;
	for (int i = 0; i < size; i++) for (int j = 0; j < size; j++) {
		points[idx][0] = i + (j % 2) * 0.5;
		points[idx][1] = j * sqrt(3) / 2;
		idx++;
	}
	int tris[TRIS][3];
	int tcnt = 0;
	for (int i = 0; i < size - 1; i++) for (int j = 0; j < size - 1; j++) {
		int v0 = i * size + j, v1 = (i + 1) * size + j, v2 = i * size + (j + 1), v3 = (i + 1) * size + (j + 1);
		tris[tcnt][0] = v0; tris[tcnt][1] = v1; tris[tcnt][2] = v2; tcnt++;
		tris[tcnt][0] = v1; tris[tcnt][1] = v3; tris[tcnt][2] = v2; tcnt++;
	}
	double total_action = 0;
	for (int v = 0; v < tot_pts; ++v) {
		// interior only
		double def = compute_deficit(points, tris, tcnt, v);
		if (isnan(def) || fabs(def) < 1e-6) continue;
		// approximate area per vertex
		double area = 0;
		for (int t = 0; t < tcnt; ++t) {
			int i = tris[t][0], j = tris[t][1], k = tris[t][2];
			if (i == v || j == v || k == v) {
				area += compute_triangle_area(points[i][0], points[i][1],
											  points[j][0], points[j][1],
											  points[k][0], points[k][1]) / 3.0;
			}
		}
		total_action += area * def;
	}
	printf("Regge action (triangular mesh): %.6f\n", total_action);
}

// ========== Discrete U(1) curvature =============
void discrete_em() {
	double theta[MAX_N][MAX_N][2]; // horizontal(0) and vertical(1) phases
	// random small phases
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
		theta[i][j][0] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
		theta[i][j][1] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
	}
	double total_F2 = 0;
	for (int i = 0; i < N - 1; i++) for (int j = 0; j < N - 1; j++) {
		double F = theta[i][j][0] + theta[i + 1][j][1] - theta[i][j + 1][0] - theta[i][j][1];
		total_F2 += F * F;
	}
	printf("Discrete EM action: %.6f\n", total_F2);
}

int main() {
	srand(time(NULL));
	printf("=== RG Flow Exponents ===\n"); rg_flow();
	printf("=== Regge Action ===\n"); regge_triangular();
	printf("=== Discrete EM ===\n"); discrete_em();
	_getch();
	return 0;
}
