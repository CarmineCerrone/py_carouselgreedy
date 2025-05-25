import networkx as nx
from py_carouselgreedy import carousel_greedy
import random

def compute_components_nx(active_labels, G):
    """
    Computes the number of connected components in the subgraph induced
    by the active labels.

    G is a networkx graph whose edges have a 'label' attribute.
    Only edges whose label is in active_labels are included in the subgraph.
    """
    # Create a subgraph with the same nodes as G
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    # Add only those edges whose label is active.
    for u, v, data in G.edges(data=True):
        if data.get("label") in active_labels:
            H.add_edge(u, v)
    # Compute and return the number of connected components.
    return nx.number_connected_components(H)


def mlst_greedy(cg_instance,solution, candidate):
    """
    Greedy function for the Minimum Label Spanning Tree (MLST) problem using networkx.

    It evaluates a candidate label by computing the number of connected components in the
    subgraph induced by the active labels (current solution ∪ {candidate}).
    The function returns the negative number of components so that a candidate leading to a
    more connected graph (fewer components) gets a higher score.
    """
    # Build the set of active labels: current solution + candidate.
    active_labels = set(solution)
    active_labels.add(candidate)

    G = cg_instance.data
    comp = compute_components_nx(active_labels, G)
    return -comp


def mlst_test_feasibility(cg_instance, solution):
    """
    Feasibility function for the MLST problem using networkx.

    A solution (i.e., a set of labels) is feasible if the subgraph induced by those labels
    is connected (i.e., it has exactly one connected component).
    """
    active_labels = set(solution)
    G = cg_instance.data
    comp = compute_components_nx(active_labels, G)
    return comp == 1


def generate_graph(num_nodes, edge_prob, num_labels, seed):
    label_pool = [f"{i}" for i in range(num_labels)]

    # Generate a random Erdos-Rényi graph
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=42)

    # Remove isolated nodes to ensure a meaningful problem
    G.remove_nodes_from(list(nx.isolates(G)))

    # Assign random labels to each edge
    for u, v in G.edges():
        G[u][v]['label'] = random.choice(label_pool)

    # Optional: ensure graph is connected (if needed for your test)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G


if __name__ == '__main__':
    # Parameters
    num_nodes = 1000
    edge_prob = 0.1
    num_labels = 5
    seed=42

    G = generate_graph(num_nodes,edge_prob,num_labels,seed)


    # Get unique labels in the graph
    candidate_elements = list({data.get("label") for _, _, data in G.edges(data=True)})

    # Initialize the CarouselGreedy instance:
    cg = carousel_greedy(
        test_feasibility=mlst_test_feasibility,
        greedy_function=mlst_greedy,
        data=G,
        candidate_elements=candidate_elements
    )

    # Run the optimization to obtain three solutions:
    # [best_solution, greedy_solution, cg_solution]
    best_solution = cg.minimize(alpha=10, beta=0.1)
    cg_solution = cg.cg_solution
    greedy_solution = cg.greedy_solution

    print("Best MLST Solution:", best_solution, "Size : ", len(best_solution))
    print("Greedy solution:", greedy_solution, "Size : ", len(greedy_solution))
    print("CG solution:", cg_solution, "Size : ", len(cg_solution))
