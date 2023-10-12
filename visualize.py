import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def visualize(data: pd.DataFrame):
    G = nx.from_pandas_edgelist(
        data,
        source="subject",
        target="object",
        edge_attr="relation",
        edge_key="relation",
        create_using=nx.MultiDiGraph()
    )

    plt.ion()
    plt.figure(figsize=(32, 32))
    nx.draw_networkx(G, 
                     with_labels=True, 
                     pos=nx.spring_layout(G))
    # nx.draw_networkx_edge_labels(G,
    #                              edge_labels=...,
    #                              pos=nx.spring_layout(G))
    plt.show()

    