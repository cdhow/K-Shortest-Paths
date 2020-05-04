#include <iostream>
#include <vector>
#include <fstream>
#include <cfloat>

/**
 * Algorithm
 * Part 1
 * - Reverse the graph directions
 * - Run Dijkstra's algorithm with dest as src to
 *   get the distances from each node to the dest,
 *   this value we be used as our heuristic
 * - Reverse the directions to their original state
 *   (we won't actually reverse them we will just store
 *   the original order and use that for part 2)
 *
 * Part 2
 * - Perform A* with the values calculated above as the heuristic
 * - Set the heuristic of the node that came before the end node to infinity
 * - Perform A* - this gives our shortest path
 * - Perform A* for each node's heuristic in that previous path equalling infinity
 * - take the shortest path from that calculation
 * - Repeat this process K times on the new shortest path
 */

// Loads the data from the specified input input file
// into an adjacency matrix
std::vector<std::vector<double>> load_matrix(std::string &infile)
{
    std::ifstream f(infile);
    if (!f.good()) std::cerr << "File does not exist: " + infile;
    std::string line;

    // Number of vertices and number of edges
    int n_v = 0, n_e = 0;

    // First line contains the n_v and n_e
    f >> n_v >> n_e;

    // Adjacency matrix of nodes from the input file
    std::vector<std::vector<double>> matrix(n_v, std::vector<double>(n_v, 0));

    // node a, node b, weight of edge between them]
    int a = 0, b = 0;
    double w = 0;

    while(std::getline(f,line)) {
        f >> a >> b >> w;
        matrix[a][b] = w;
    }
    return matrix;
}

// Takes a matrix as a param and return a transposed copy of that matrix
std::vector<std::vector<double>> transpose_matrix(std::vector<std::vector<double>> &matrix)
{
    int vertices = matrix.size();
    std::vector<std::vector<double>> transpose(vertices, std::vector<double>(vertices));
    for (int i = 0; i < vertices; i++)
        for (int j = 0; j < vertices; j++) {
            transpose[j][i] = matrix[i][j];
        }
    return transpose;
}

// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
double minDistance(std::vector<double> &dist, std::vector<int> &p)
{
    // Initialize min value
    double min = DBL_MAX;
    int min_index;

    for (int v = 0; v < dist.size(); v++)
        if (!p[v] && dist[v] <= min)
            min = dist[v], min_index = v;

    return min_index;
}

// The Dijkstra's single source shortest path algorithm
// it takes the transposed matrix, where the directions are
// reversed from the original matrix, this allows for us to
// set the original destination as the source and in turn retrieve
// the distance from every node to the destination
std::vector<double> dijkstra(std::vector<std::vector<double>> &matrix, int src)
{
    // Output vector , dist[i] will hold the shortest path
    // cost from src to node i
    std::vector<double> dist(matrix.size(), DBL_MAX);

    // Unit vector, if node i is included in the shortest path or
    // the shortest distance from src to i is processed then
    // p[i] = 1 else = 0
    std::vector<int> p(matrix.size(), 0);

    // Distance of src node from itself is always 0
    dist[src] = 0.0;

    // Find shortest path for all vertices
    for (int count = 0; count < matrix.size() - 1; count++) {
        // Pick the minimum distance vertex from the set of vertices not
        // yet processed
        double min_v = minDistance(dist, p);

        // Mark the node as processed
        p[min_v] = true;

        // Update the dist value of the nodes adjacent to the processed node
        for (int v = 0; v < matrix.size(); v++) {
            // Update dist[v] only if it is not in p, there is an edge from
            // min_v to v, and total weight of path from src to  v through m is
            // smaller than the current value of dist[v]
            if (!p[v] && matrix[min_v][v] && dist[min_v] != DBL_MAX
                && dist[min_v] + matrix[min_v][v] < dist[v])
                dist[v] = dist[min_v] + matrix[min_v][v];
        }
    }

    // Return the list of distances from src to all nodes
    return dist;
}

// TODO: A*
std::vector<int> astar(std::vector<std::vector<int>> &matrix, std::vector<int> &heuristic)
{

}

// Takes a make
void k_shortest_path(std::vector<std::vector<int>> &matrix, std::vector<int> &heuristic, const int &k)
{
    /*
     * For 1 to K:
     *      path = astar(matrix, heuristic)
     *
     *      // Run astar on path for each node equalling infinity
     *      // keep track of the min path
     *      For node in path:
     *          store node_heuristic = heuristic[node]
     *          update it to infinity
     *          heuristic[node] = infinity
     *          current_path = astar(matrix, herustic)
     *          if (current_path_cost < path_cost)
     *              path = current_path
     *              path_cost current_path_cost
     *
     *          // revert the heuristic to it's orignal vaule
     *          heuristic[node] = ode_heuristic
     *
     *      // The path will be the new second shortest
     *      // TODO: need a way to check that second path to third path wont produce first path
     *
     */
}


int main() {
    int src = 4;
    int dst = 2007;

    std::string infile = "input_data.txt";

    // Get the adjacency matrix
    std::vector<std::vector<double>> matrix = load_matrix(infile);

    // Get a transposed copy of the matrix,
    // a transposed matrix is just the original
    // matrix with all directions reversed
    std::vector<std::vector<double>> matrix_t = transpose_matrix(matrix);

    // We use Dijkstra's algorithm to with a transposed matrix
    // and the src as the original destination
    // this will give us the shortest distance that every node is
    // from the destination, we will used these distances as our heuristics
    std::vector<double> heuristic = dijkstra(matrix_t, dst);

    for (auto h : heuristic)
        std::cout << h << "\n";
    return 0;
}
