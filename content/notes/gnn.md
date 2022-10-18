---
author: "Kush Kothari"
title: "An Intro to Graph Machine Learning"
date: 2022-10-18T14:56:19+05:30
draft: false
tags:
    - graphs
---

Graph data is quite abundant around us. Almost all data that we can think of can be represented as a graph in some way or the other. Apart from the obvious applications of graph research in GPS and the analysis of social networks, we can also model a lot of other data in graphs. For example, images can be thought of as just nodes with RGB features connected as a grid. NLP research also makes a lot of graph data for example parse trees etc. Graph ML, as a whole is applied in a large variety of domains like:
- Physics: Detecting and plotting the course of particles in a collider experiment.
- Biology: Using gene analysis and GNNs to predict the possibility of a patient responding to a certain kind of statement.
- Chemistry: Used by pharmacists to model drug-drug, drug-protein and protein-protein interactions to understand how effective or detrimental a substance can be to one's body.

# Graph Embedding Methods
Graph Embedding Methods deal with representing the nodes of a graph as vectors in multidimensional vector space. This is called a **latent** representation, because it itself may not hold a lot of obvious meaning, but it encodes the relationships in the graph well. The given outputs are in continuous vector space can be easily explored and exploited using existing statistical methods.

## Dealing with Sparsity while creating Graph Embeddings
A graph can either be considered as dense (a lot of edges as compared to the nodes) or sparse (a lot less edges as compared to nodes). Analyzing sparse graphs is easier when we are trying to apply combinatorial techniques. However, it doesn't work well when we try to use statistical techniques on sparse graphs. This problem will have to be dealt with if we are trying to learn an embedding for a graph.

## Learning Social representations
Graph Embeddings seek to learn latent representations of social interactions in the graph. For this they must have the following qualities.
1. Adaptability: This relates to the ease of training a latent representation. Newer nodes or edges should not trigger a complete retraining of the embedding.
2. Community Aware: The distance between nodes though the latent dimensions should represent a metric for evaluating social similarity between them. This allows for easier generalisation of the networks and better results when statistical methods are applied on the resultant representations.
3. Low-dimensional: The lower the number of dimensions in the representation, the easier it will be to make inferences and converge on a final representation.
4. Continuous: The resultant representation must be over continuous space thus allowing for smoother decision boundaries in downstream tasks.

## Random Walks
For creating embeddings we need to transform the graph data into something that can effectively stream the structure of the graph (and hence it's short-term and long-term complexities). Short random walks are perfect for this. Shorter term complexities are captured well by the actual structure and orderings of the nodes in the walk. Longer term complexities are captured through the probability distribution from which the initial node is sampled.

Methods that use **short random walks** have 2 particular advantages:
- These are trivially easy to parallelize. Several parallel computational resources (like threads, processes or machines) can be set up to explore different sections of the graph parallelly.
- Since the walks are short, only a small subset of nodes will be updated at a time. This makes it easy to accomodate small changes to an existing embedding without the need for global recomputation.

## Power Law and how a graph embedding is equivalent to a language embedding
The power law can be simply stated as:
> A functional relationship between 2 quantities, where a relative change in one quantity results in a proportional relative change in another quantitiy.

The power law distribution has the mathematical form of:
$$y = kx^\alpha$$

Now, it can be mathematically proved that if the degree-distribution of a graph follows a power-law, the frequency with which various vertices will appear in short random walks will also follow a power law. Additionally, from past research we know that in natural language, the frequency with which words appear in natural language also follows a similar power law. This is shown in the diagram below.

{{<figure src="../static/power-law-deepwalk.png" caption="Figure 2. DeepWalk: Online Learning of Social Representations">}}

This essentially means that existing language modelling techniques that are already qualified to deal with data that follows a power distribution will also generalize to finding social representations in graphs.

Citation: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)

