#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <cfloat>
#include <memory>

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
 *
 * Note: I would usually split this program into multiple files, but
 * for simplicity for the marker I'll keep it all in the single file
 */


// Loads the data from the specified input input file
// into an adjacency matrix, returns a tuple of, (src, dst, matrix)
std::tuple<int, int, std::vector<std::vector<double>>> load_matrix(std::string &infile)
{
    std::ifstream f;
    f.open(infile);
    if (!f.good()) std::cerr << "File does not exist: " + infile;
    std::string line;

    // Number of vertices and number of edges
    int n_v = 0, n_e = 0;

    // First line contains the n_v and n_e
    f >> n_v >> n_e;

    // Adjacency matrix of nodes from the input file
    std::vector<std::vector<double>> matrix(n_v, std::vector<double>(n_v, 0));

    // node a, node b
    int a = 0, b = 0;
    // weight of edge between a, b
    double w = 0;
    // source and destination nodes
    int src = 0, dst = 0;

    int lineNum = 1;
    while(lineNum <= n_e+1) {
        std::getline(f,line);
        f >> a >> b >> w;
        matrix[a][b] = w;
        lineNum++;
    }

    // Last line contains src and dst
    f >> src >> dst;
    f.close();

    return std::make_tuple(src, dst, matrix);
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


/*
 * ###################### This region is for part 2 ###############################
 */


// A node (state) that will store the it's path
// and cost function for the A* algorithm
class Node {
private:
    int id;

    // Cost function value. i.e g(n) = f(n) + h(n),
    // where f(n) = pathCost and h(n) = heuristic
    double g{0};

    // Current path cost from source node
    // to this node
    double pathCost{0};

    // Pointer to parent node
    Node *parent{nullptr};
public:
    Node() : id{0} {}
    explicit Node(const int &id) :id{id} {}


    int getID() const { return id; }

    double getPathCost() const { return pathCost; }

    double getG() const { return g; }

    Node* getParent()
    {
        return parent;
    }

    // Update the path cost and the g-value
    void update_costs(const double &pCost, const double &heuristic)
    {
        pathCost = pCost;
        g = pCost + heuristic;
    }

    // Set the parent node
    void update_parent(Node *p)
    {
        parent = p;
    }

};

// Functor struct to order a priority queue bt a Node's g - value
struct LessThanByG
{
    bool operator()(const Node* lhs, const Node* rhs)
    {
        return lhs->getG() < rhs->getG();
    }
};


// TODO: A*
std::pair<double, std::vector<int>> astar(
        std::vector<std::vector<double>> &matrix,
        std::vector<double> &heuristic,
        const int &src,
        const int &dest)
{

    // Vector that contains All node objects
    std::vector<Node*> nodes;

    // Populate nodes vector
    // Nodes are labelled 0 to number of nodes
    nodes.reserve(matrix.size());
    for (int i=0; i< matrix.size(); i++)
        nodes.push_back(new Node(i));

    // Priority queue to contain node pointers, ordered by the node's g-value
    std::priority_queue<Node*, std::vector<Node*>, LessThanByG> pq;

    // Push the source node to the qQueue
    nodes[src]->update_costs(0, heuristic[src]);
    pq.push(nodes[src]);

    // Run A*
    Node *current;
    while (!pq.empty())
    {
        // Pop node
        current = pq.top();
        pq.pop();

        int id = current->getID();

        // If we reach the destination
        if (id == dest)
            break;

        // Push children
        for (int i=0; i<nodes.size(); i++) {
            if (matrix[id][i] != 0) {
                // Child is found in matrix
                // Update path cost to child and G-value
                double costToChild = nodes[i]->getPathCost() + current->getPathCost();
                nodes[i]->update_costs(costToChild, heuristic[i]);
                // Link the parent to the child
                nodes[i]->update_parent(current);
                pq.push(nodes[i]);
            }
        }
    }

    // Path cost
    double pCost = current->getPathCost();

    // Traverse the parent list to obtain the path
    std::vector<int> path;
    while (current->getParent() != nullptr) {
        path.push_back(current->getID());
        current = current->getParent();
    }

    return std::make_pair(pCost, path);
}

// Takes a make
void k_shortest_path(std::vector<std::vector<double>> &matrix,
                    std::vector<double> &heuristic,
                    const int &k,
                    const int &src,
                    const int &dest)
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



    // Run A* to get the initial shortest path
    std::vector<int> kShortestPath;
    double kShortestPathCost;
    std::tie(kShortestPathCost, kShortestPath)= astar(matrix, heuristic, src, dest);

    // Print initial path
    std::cout << "k=" << k << ": " << kShortestPathCost << std::endl;

    // For K-iterations
    for (int i=1; i<k; i++)
    {
        // For each node in the last shortest path we will run
        // A* with that nodes edge disabled (where heuristic=infinity).
        // We will take the best of those paths as the kth shortest
        // and permanently make that nodes heuristic=infinity for
        // the remaining k-1 iterations
        // This is the node that when the heuristic = infinity, we get the shortest path
        double minPathCost = DBL_MAX;
        std::vector<int> minPath;
        int nodeWithEffect;
        for (int node : kShortestPath) {
            // Store the heuristic so we can revert the changes later
            double prev_h = heuristic[node];

            // Increase heuristic to max to stop A* from choosing that path
            heuristic[node] = DBL_MAX;

            // Run A* with the update heuristic
            std::vector<int> currentPath;
            double currentPathCost;
            std::tie(currentPathCost, currentPath) = astar(matrix, heuristic, src, dest);

            // Check if that path cost is less than the last
            if (currentPathCost < minPathCost) {
                // Update the shortest
                minPath = currentPath;
                minPathCost = currentPathCost;
                nodeWithEffect = node;
            }

            // Revert the heuristic to it original state
            heuristic[node] = prev_h;
        }

        // Shortest kth path is the min_path
        kShortestPath = minPath;
        // Update the nodeWithEffect heuristic so it is
        // not traversed next iteration
        heuristic[nodeWithEffect] = DBL_MAX;

        // Print the Kth cost
        std::cout << "k=" << i << ": " << kShortestPathCost << std::endl;

    }
}


int main() {
    std::string infile = "input_data.txt";

    // Get the adjacency matrix, the k-value, and the src and dest nodes
    std::vector<std::vector<double>> matrix;
    int src, dst;
    std::tie(src, dst, matrix) = load_matrix(infile);

    // Get a transposed copy of the matrix,
    // a transposed matrix is just the original
    // matrix with all directions reversed
    std::vector<std::vector<double>> matrix_t = transpose_matrix(matrix);

    // We use Dijkstra's algorithm to with a transposed matrix
    // and the src as the original destination
    // this will give us the shortest distance that every node is
    // from the destination, we will used these distances as our heuristics
    std::vector<double> heuristic = dijkstra(matrix_t, dst);

    int k = 3;
    // Run the K-Shortest path algorithm
    k_shortest_path(matrix, heuristic, k, src, dst);



    return 0;
}
