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

{{<figure src="../static/deepwalk-banner.png" caption="Figure 1. DeepWalk: Online Learning of Social Representations">}}

## Dealing with Sparsity while creating Graph Embeddings [^1]
A graph can either be considered as dense (a lot of edges as compared to the nodes) or sparse (a lot less edges as compared to nodes). Analyzing sparse graphs is easier when we are trying to apply combinatorial techniques. However, it doesn't work well when we try to use statistical techniques on sparse graphs. This problem will have to be dealt with if we are trying to learn an embedding for a graph.

## Learning Social representations [^1]
Graph Embeddings seek to learn latent representations of social interactions in the graph. For this they must have the following qualities. [^1]
1. Adaptability: This relates to the ease of training a latent representation. Newer nodes or edges should not trigger a complete retraining of the embedding.
2. Community Aware: The distance between nodes though the latent dimensions should represent a metric for evaluating social similarity between them. This allows for easier generalisation of the networks and better results when statistical methods are applied on the resultant representations.
3. Low-dimensional: The lower the number of dimensions in the representation, the easier it will be to make inferences and converge on a final representation.
4. Continuous: The resultant representation must be over continuous space thus allowing for smoother decision boundaries in downstream tasks.

## Random Walks [^1]
For creating embeddings we need to transform the graph data into something that can effectively stream the structure of the graph (and hence it's short-term and long-term complexities). Short random walks are perfect for this. Shorter term complexities are captured well by the actual structure and orderings of the nodes in the walk. Longer term complexities are captured through the probability distribution from which the initial node is sampled. [^1]

Methods that use **short random walks** have 2 particular advantages:
- These are trivially easy to parallelize. Several parallel computational resources (like threads, processes or machines) can be set up to explore different sections of the graph parallelly.
- Since the walks are short, only a small subset of nodes will be updated at a time. This makes it easy to accomodate small changes to an existing embedding without the need for global recomputation.

## Power Law and how a graph embedding is equivalent to a language embedding [^1]
The power law can be simply stated as:
> A functional relationship between 2 quantities, where a relative change in one quantity results in a proportional relative change in another quantitiy.

The power law distribution has the mathematical form of:
$$y = kx^\alpha$$

Now, it can be mathematically proved that if the degree-distribution of a graph follows a power-law, the frequency with which various vertices will appear in short random walks will also follow a power law. Additionally, from past research we know that in natural language, the frequency with which words appear in natural language also follows a similar power law. This is shown in the diagram below.

{{<figure src="../static/power-law-deepwalk.png" caption="Figure 2. DeepWalk: Online Learning of Social Representations">}}

This essentially means that existing language modelling techniques that are already qualified to deal with data that follows a power distribution will also generalize to finding social representations in graphs.

## Kullback-Leibler divergence [^2]
> In mathematical statistics, the KL divergence (also called relative entropy), denoted as:
> $$D_{KL}\left( P  \Vert  Q \right)$$
> is a type of statistical distance: a measure of how one probability distribution P is different from a second reference probability distribution Q.

A simple interpretetion of the KL divergence of P from Q is the expected excess surprise from using Q as a model when the actual distribution is P. Here is a great video to get intution for the formula for KL Divergence: [Intuitively understanding KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM)

Formula for KL Divergence:
$$D_{KL}\left(P \middle\Vert Q\right)= \sum_{i} P(i)  log\frac{P(i)}{Q(i)}$$
$$D_{KL}\left(P \middle\Vert Q\right)= \int P(x)  log\frac{P(x)}{Q(x)}dx$$

## Cross Entropy Loss
This is the most famous loss that we used to calculate the difference between the probability distribution of a model given parameters θ, and the true probability distribution that we wanted our model to predict. You might wonder why we use something like cross-entropy loss to compare distance between the two distributions (as explained above). In this section we prove why changing θ to minimize the cross-entropy loss is equivalent to minimizing the KL Divergence between the distributions.

Let us first quantify the 2 distributions in question:
1. For input image: \\(x_{i}\\)
2. We have the true probability of the output as: \\(P^{\*}(y \vert  x_{i})\\)
3. We have the probability predicted by our model as: \\(P(y \vert  x_{i},\theta)\\)

Now our aim in machine learning is to find θ such that we minimize the KL Divergence of P\* and P. In other words, we want:
$$\underset{\theta}{argmin}\  D_{KL}\left(P^{\*}\ \middle\Vert \ P\right)$$

Expanding as,

$$
\begin{aligned}
D_{KL}\left(P^{\*}\ \middle\Vert \ P\right) & = \sum_{y\ \in\ classes} P^{\*}(y \vert x_{i})\ \log\frac{P^{\*}(y \vert x_{i})}{P(y \vert\ x_{i},\ \theta)}\\\\
& = \underbrace{\sum_{y\ \in\ classes} P^{\*}(y \vert x_{i})\ \log P^{\*}(y \vert x_{i})}\_{independent\ of\ \theta} - \sum_{y\ \in\ classes} P^{\*}(y \vert x_{i})\ \log P(y \vert\ x_{i},\ \theta)
\end{aligned}
$$

Thus we can see that:

$$D_{KL}\left(P^{\*}\ \middle\Vert \ P\right) = - H\left(P^{\*}\right) + H\left(P^{\*},P\right)$$

Thus the value of θ that minimizes the cross-entropy between P\* and P, also minimizes the KL Divergence between P\* and P.

$$\underset{\theta}{argmin}\ D_{KL}\left(P^{\*}\ \middle\Vert \ P\right) \equiv \underset{\theta}{argmin}\ H\left(P^{\*},\ P\right)$$

## Word Embeddings? Why though? [^3]
For a second, pause and just imagine how we would explain words and their meanings to computers. How would you provide the following association to a computer?
> Queen is to King as Woman is to Man

Perhaps, if you had a mathematical function that takes in words and represents them as vectors, you could say:
$$\Phi("King") = \Phi("Queen") - \Phi("Woman") + \Phi("Man")$$

In word embeddings, the aim is to capture such relations between words mathematically.

Often you will also see embeddings being writtien as **distributed representations** of words. Why is that?
> This is because we are distributing the actual meaning of the word over multiple mathematical dimensions. The quality of a "King" being a man is represented by a subset of the dimensions of the vector representation. And, the a particular dimension of the vector representation represents a subset of the qualities a word like "King" can posses.

## An Intro to Language Modelling [^1]
The fundamental goal of language modelling is to estimate the likelihood of a specific sequence of words appearing in a corpus. More formally, given any sequence of words (w1,w2,w3,...wn) from the training corpus, we would like to maximize:
$$Pr\left(w_{n} \middle\vert w_{1},w_{2},w_{3},...,w_{n-2},w_{n-1}\right)$$
over the entire training corpus.

But if we are representing words with the corresponding distributed vector representation, we can also maximize:
$$Pr\left(w_{n} \middle\vert \Phi(w_{1}),\Phi(w_{2}),\Phi(w_{3}),...,\Phi(w_{n-2}),\Phi(w_{n-1})\right)$$

But recent advancements in Language modelling turn this problem around in ways that only use the "word-vectorizer" function \\(\Phi\\) only once. We now aim to do,
$$\underset{\Phi}{argmin}\ -\log Pr\left(\\{ v_{i-w},...,v_{i-1},v_{i+1},...,v_{i+w}\\}\ \middle\vert\ \Phi(v_{i})\right)$$

Do you understand how this model is "order-independant"? Hint: Check for set notation somewhere.

## The curse of dimensionality [^4]
In a normal probabilistic model of a language having a vocabulary of \\(\lvert\mathcal{V}\rvert = 1,000,000\\), how many parameters would you need to create a language model to predict the joint distribution of 10 words?

If you think about it, this is an extremely large number and it doesn't seem feasible to make a model like that. Neither can you share any of the values, because all parameters are discrete and it is difficult to come up with an efficient method of sharing these values. But unlike simplifying a discrete model, simplifying a continuous model is way easier, because continuous functions can be always be approximated using simpler (smoother functions). Thus we must find some way of representing discrete words in a continuous domain. Let us build a neural network to do this for us.

## A Neural Probabilistic Model [^4]
{{<figure src="../static/probabilistic-language-model.png" caption="Figure 1. A Neural Probabilistic Language Model">}}
This was **probably** the first model to use deep learning to learn word representations. But the output layer had \\(\lvert\mathcal{V}\rvert\\)
$$ y = b + Wx + U\tanh\left(d + Hx\right)$$
$$ \hat{P}\left(w_{t} \vert w_{t-1}, w_{t-2}, w_{t-3}, \ldots w_{t-n+1} \right) = \frac{e^{y_{w_{t}}}}{\sum_{i} e^{y_{i}}}$$

## The SkipGram Model [^5]
  SkipGram is a language model that maximizes the co-occurence probability among the words that appear in a window \\(\mathcal{w}\\) around the current word. In the simple softmax version of Word2Vec, we update the representation vectors and the probability function, using SGD.
{{<figure src="../static/skipgram-algorithm.png" caption="Algorithm 2. DeepWalk: Online Learning of Social Representations">}}

But how do you actually find \\(\- \log Pr\left(u_{k} \vert \Phi(v_{j})\right)\\)?

[^5] describes \\(p(\mathcal{w}\_{t+j} \vert \mathcal{w}\_{t})\\) as

$$
p(w_{O}\vert w_{I}) = \frac{\exp \left(v_{\mathcal{w}\_{O}}^{\prime T} v_{\mathcal{w}\_{I}} \right)}{\sum_{\mathcal{w}=1}^W \exp \left(v_{\mathcal{w}}^{\prime T}v_{\mathcal{w}\_{I}}\right)}
$$

How feasible do you think this is?

[^1]: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
[^2]: [Kullback–Leibler divergence Wikipedia](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
[^3]: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
[^4]: [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
[^5]: [Distributed Representations of Words and Phrases and their Compositionality](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
