import networkx as nx
import logging
from py_carouselgreedy import carousel_greedy

# Configure the logger to display INFO-level messages
logging.basicConfig(level=logging.INFO)


# Feasibility function for the Vertex Cover problem.
# A solution is feasible if every edge in the graph has at least one endpoint in the solution set.
def my_feasibility(cg_instance, solution):
    graph = cg_instance.data
    for (u, v) in graph.edges():
        if u not in solution and v not in solution:
            return False
    return True


# Greedy function for the Vertex Cover problem.
# This function evaluates a candidate node by counting how many currently uncovered edges it would cover if added to the solution.
def my_greedy(cg_instance, solution, candidate):
    graph = cg_instance.data
    uncovered = 0
    for (u, v) in graph.edges():
        if u not in solution and v not in solution:
            if candidate == u or candidate == v:
                uncovered += 1
    return uncovered


def main():
    # Generate an Erdos-Rényi graph with 50 nodes and edge probability 0.1
    n = 50
    p = 0.1
    seed = 42
    G = nx.erdos_renyi_graph(n, p, seed=seed)


    # The list of candidate elements consists of all nodes in the graph
    candidate_elements = list(G.nodes())

    # Create an instance of Carousel Greedy for the vertex cover problem.
    cg = carousel_greedy(
        test_feasibility=my_feasibility,
        greedy_function=my_greedy,
        data=G,
        candidate_elements=candidate_elements,
    )

    best_solution = cg.minimize(alpha=10, beta=0.1)
    cg_solution = cg.cg_solution
    greedy_solution = cg.greedy_solution
    print("Best Vertex Cover:", best_solution, "Size : ",len(best_solution))
    print("Greedy solution:", greedy_solution, "Size : ",len(greedy_solution))
    print("CG solution:", cg_solution, "Size : ",len(cg_solution))


if __name__ == '__main__':
    """
    Basic example of how to use the Carousel Greedy (CG) library on the Vertex Cover problem.

    This script generates a random Erdos-Rényi graph and applies the CG algorithm to find 
    a feasible vertex cover.

    Intended as a minimal working example to demonstrate usage of the CG library.
    
    For more advanced configurations, see the 'mvc_enhanced.py' example.
    """
    main()