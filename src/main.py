import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx

from models.observer import Observer
from .train import *

data = {
    'Box Office Sales': [85.1, 106.3, 50.2, 130.6, 54.8, 30.3, 79.4, 91.0, 135.4, 89.3],
    'Production Costs': [8.5, 12.9, 5.2, 10.7, 3.1, 3.5, 9.2, 9.0, 15.1, 10.2],
    'Promotion Costs': [5.1, 5.8, 2.1, 8.4, 2.9, 1.2, 3.7, 7.6, 7.7, 4.5],
    'Book Sales': [4.7, 8.8, 15.1, 12.2, 10.6, 3.5, 9.7, 5.9, 20.8, 7.9]
}

G = nx.Graph()

class Console(Observer):
  def update(self, *args, **kwargs):
    # print(kwargs["event"], kwargs["a"], kwargs["b"])
    pass

class Graph(Observer):
  previous = None
  def update(self, *args, **kwargs):
    if "terminal_node" in kwargs:
      # print(kwargs["terminal_node"])
      pass
    
    if "event" in kwargs and kwargs["event"] == "graph":
      G.add_node(kwargs["id"], color="dimgray", size=10)
      if self.previous is not None:
        G.add_edge(kwargs["id"], self.previous)
      self.previous = kwargs["id"]

# The following line will create a list of data points. It does this by:
# (1) creating a list containing each of the lists in `data`
# (2) unpacking the lists from (1) and passing them into `zip`
# (3) using `zip` to wrap the elements from each list together in tuples
# (4) using the elements output from `zip` to create a list
data_set = list(zip(*[data[key] for key in data]))

train = Train()

if __name__ == "__main__":
  graph = Graph()
  Number.observers = [graph]
  Train.observers = [graph]
  model = train.create_model(4)
  losses = []

  for _ in range(10):
      G.clear()
      losses.append(train.train_epoch(model, data_set, train.l2_loss))
      pass

  fig, ax = plt.subplots(figsize=(8,4))

  # ax.plot(losses)
  # ax.set_yscale('log')
  # ax.set_ylabel('Loss')
  # ax.set_xlabel('Training Step');
  # ax.grid(True)

  node_colors = [G.nodes[n]["color"] for n in G.nodes]
  node_sizes = [G.nodes[n]["size"] for n in G.nodes]

  nx.draw(
      G, 
      node_color=node_colors,
      node_size=node_sizes,
      font_color="white",
      font_weight="bold"
  )

  plt.show()