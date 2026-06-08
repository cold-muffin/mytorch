import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx

from .train import *

data = {
    'Box Office Sales': [85.1, 106.3, 50.2, 130.6, 54.8, 30.3, 79.4, 91.0, 135.4, 89.3],
    'Production Costs': [8.5, 12.9, 5.2, 10.7, 3.1, 3.5, 9.2, 9.0, 15.1, 10.2],
    'Promotion Costs': [5.1, 5.8, 2.1, 8.4, 2.9, 1.2, 3.7, 7.6, 7.7, 4.5],
    'Book Sales': [4.7, 8.8, 15.1, 12.2, 10.6, 3.5, 9.7, 5.9, 20.8, 7.9]
}

# The following line will create a list of data points. It does this by:
# (1) creating a list containing each of the lists in `data`
# (2) unpacking the lists from (1) and passing them into `zip`
# (3) using `zip` to wrap the elements from each list together in tuples
# (4) using the elements output from `zip` to create a list
data_set = list(zip(*[data[key] for key in data]))

if __name__ == "__main__":
  model = create_model(4)
  losses = []
  for _ in range(1000):
      losses.append(train_epoch(model, data_set, l2_loss))

  fig, ax = plt.subplots(figsize=(8,4))

  ax.plot(losses)
  ax.set_yscale('log')
  ax.set_ylabel('Loss')
  ax.set_xlabel('Training Step');
  ax.grid(True)

  # G = nx.Graph()

  # G.add_node("A", color="dimgray", size=300)
  # G.add_node("B", color="dimgray", size=300)
  # G.add_node("C", color="dimgray", size=300)

  # G.add_edge("A", "B")
  # G.add_edge("B", "C")

  # G.nodes["B"]["color"] = "red" 
  # G.nodes["B"]["size"] = 700

  # node_colors = [G.nodes[n]["color"] for n in G.nodes]
  # node_sizes = [G.nodes[n]["size"] for n in G.nodes]

  # nx.draw(
  #     G, 
  #     with_labels=True, 
  #     node_color=node_colors, 
  #     node_size=node_sizes,
  #     font_color="white",
  #     font_weight="bold"
  # )

  plt.show()