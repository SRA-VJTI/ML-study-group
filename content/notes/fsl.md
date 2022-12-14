---
author: "Kush Kothari"
title: "A Survey of Techniques used in Few Shot Learning"
date: 2022-11-05T14:56:19+05:30
draft: false
tags:
    - few shot learning
---

Survey Citation: Generalizing from a Few Examples: A Survey on Few-Shot Learning[^1]
## Defining FSL
A very famous definition of **Machine Learning is**:
> A machine is said is said to learn from experience \\(\mathit{E}\\) with respect to some task \\(\mathit{T}\\) and performance measure \\(\mathit{P}\\) when its performance with \\(\mathit{E}\\) improves on \\(\mathit{T}\\) as measured by \\(\mathit{P}\\).

From this definition of ML, we can further define FSL:
> Few-Shot Learning (FSL) is a type of machine learning problem (as specified by E,T and P), where \\(\mathit{E}\\) contains only a limited number of examples with supervised information for the target \\(\mathit{T}\\).

Three of the most common tasks in Few-Shot Learning are defined below:
1. Few-Shot Classification: Here, learning classifiers are given only a few labeled examples of each class.
2. Few-Shot Regression: Few-Shot Regression estimates a regression function *h* given only a few input-output examples pairs sampled from that function.
3. Few-Shot Reinforcement Learning: Targets finding a policy given only a few trajectories consisting of state-action pairs.

Some applications where we can apply Few-Shot Learning:
1. In some ways, in order to reach human intelligence we must eventually solve FSL problems. Because we humans don't really need a lot of data to reach human level intelligence (or atleast relatively less, as compared to the current state of ML algorithms).
2. We can use FSL where the actual data that we want to classify is actually very less. An example of this the detection of disaster circumstances. We don't really have a lot of dataduring disaster circumstances, so with the limited data/examples that we do have, we need to build strong ML algorithms.
3. A lot of times, actually collecting data can be quite difficult and complex. To overcome tis complexity we might want to employ FSL.

> An important thing to note in FSL is that in order to train efficient models with limited data we turn to using *prior knowledge* in order to train stronger models. One common type of FSL method is Bayesian learning. It provides the given training set with some prior probability distribution availabe from the problem statement.

{{<figure src="../static/prior-knowledge-fsl.png" caption="Table 2. Generalizing from a Few Examples: A Survey on Few-Shot Learning">}}

> Some related terminology:
>
> One-Shot Learning: When there is only one example ofeach target class with supervised info in E, it is called one-shot learning.
>
> Zero-Shot Learning: When there are no examples of a target class with supervised info in E, this situation is called Zero-Shot Learning.

## Problems in ML relevant to FSL
We have many many classes of work in Machine Learning. Some are more specific thant the others, and some are more rlevant to FSL thant the others.

- Weakly Supervised Learning: In this problem, learning algorithms learn from E that contain incomplete/inaccurate/noisy supervised information. This can be further divided into two types:
  - Semi-supervised Learning: In this case, there are usually a few labelled examples in E and a lot of unlabelled examples.
  - Active Learning: In this case, the E starts unlabelled, but we have an option to have an **oracle** that can label selected samples on our will. This is mostly used when the actual labelling is an expensive operation (e.g. when we need annotations from a human or just a very large model which takes time to process).
- Imbalanced Learning: As was explained above, in cases like disaster or fraud detection, we have very few positive data points as compared to negative data points. But ideally the positive data points have to be given equivalent (if not more) weightage. If we take only the positive examples (and possibly a few negative examples) here, we can apply FSL Techniques to build a dependable classifier.
- Transfer Learning: This transfer knowledge from the source domain/task (where data is abundant) to a similar domain/task where data isn't really abundant. Domain Adaptation is a sub-problem of this, where the task is the same but the domains aren't.
- Meta-Learning: A meta learner is a learner trained over a range of tasks on the same dataset. This is particularly good at learning generic information about a particular domain across multiple tasks. Beyond this, a learner can specialize a meta-learner to a new task using task-specific information. This is useful when we have many tasks of a domain solved, but certain tasks remain elusive due to less data or lack of a good algorithm.

## The core problem of FSL
Given a hypothesis \\(h\\) we define an expected **Risk**, which is essentially the expected value of the loss function given the probability of the data (i.e. the joint probability of the x and y).
$$
R(h) = \int l\left(h(x),y\right)dp(x,y) = \mathbb{E}\left[l\ (h(x),y)\right]
$$
When the joint probability of the data occuring isn't provided we just calculate the emperical training loss over the given examples.
$$
R_{I}(h) = \frac{1}{I} \sum^{I}\_{i=1} l\ (h(x_{i}),\ y)
$$

Using the above functions we can define 3 kinds of hypotheses:
1. \\(\hat{h} = argmin_{h}R\ (h)\\) as the hypotheses that minimizes expected risk
2. \\(h^{\*} = argmin_{h \in \mathcal{H}}R\ (h)\\) as the hypothese in \\(\mathcal{H}\\) that minimizes expected risk.
3. \\(h_{I} = argmin_{h \in \mathcal{H}}R_{I}\ (h)\\) as the hypothese in \\(\mathcal{H}\\) that minimizes emperical risk.

As you can see, there can be two kinds of errors. One that is caused due to our selection of the hypotheses space \\(\hat{h} - h^{\*}\\), and the other caused due to us not knowing the entire input-output data distribution \\(h^{\*} - h_{I}\\). When we have less data available, the second kind of error is what increases unnecessarily and that is what we must aim to minimize.

{{<figure src="../static/hypotheses-fsl.png" caption="Figure 1. Generalizing from a Few Examples: A Survey on Few-Shot Learning">}}

## Ways of solving a FSL problem
There are three major ways in which we can tackle an FSL problem, each of which can be applied depending on the situation and the problem statement.
1. Data: When we lack data, we just augment it! These methods list use prior knowledge to augment the training set, after which standard ML models can be used to train on the data. Once the data is augmented we can somewhat guarantee a more accurate risk minimizer.
2. Model: These methods use prior knowledge from the task to constrain the complexity of \\(\mathcal{H}\\), so we have a much smaller and more accurate hypotheses space to work with. This in turn guarantees the final hypotheses will be much more accurate.
3. Algorithm: These methods integrate prior knowledge about the task into the training algorithm itself, thus finding hypotheses with lesser risk. This can be done by providing a good initialization and/or augmenting the algorithm so it uses prior knowledge.

{{<figure src="../static/3-methods-fsl.png" caption="Figure 2. Generalizing from a Few Examples: A Survey on Few-Shot Learning">}}
{{<figure src="../static/brief-overview-of-methods-fsl.png" caption="Figure 3. Generalizing from a Few Examples: A Survey on Few-Shot Learning">}}

We will now go into detail in each of the categories.

# Data
We use data augmentation to enrich \\(\mathit{E}\\) with more supervised information. With this augmentation, the data will be sufficient enough to obtain a reliable hypothesis. This probably sounds simpler than it is. Data Augmentation rules can always be hand-crafted and be used as pre-processing methods for FSL. However, designing these rules depends heavily on domain knowledge and requires expensive labour cost. Morever, it is unlikely humans will be able to find all good augmentation rules by hand.

So here, we focus on 3 advanced data augmentation methods that go beyond simple hand-crafted rules. We classify them based on the location from which the augmented data is obtained.
{{<figure src="../static/data-aug-methods-fsl.png" caption="Figure 4. Generalizing from a Few Examples: A Survey on Few-Shot Learning">}}

## Transforming samples from the training data
We use the existing training data to learn transformations between images of the same class. Then we apply the same transformations on the rest of the training data.

Review:
- An early FSL paper learns a set of geometric transformations from a similar class by iteratively aligning each sample with other samples.
- Auto-encoders can be used to learn variability across a single class and then apply the same variability to other classes.
- Add a sort of PCA along the features of data, build a set of independant attribute strength regressors from this data, and then apply these on the data.

## Adding samples from a weakly labelled or unlabelled dataset
Here we augment our existing data by adding samples from an external dataset. This can be also be unlabelled or weakly labelled, but more specific data to the our existing data is a plus. Even if the data that is added is not labelled, the new data actually helps us get a better estimate of the distribution of the input-output data.

Review:
- In some papers, a classifier (SVM, in this case) is learned from the existing training data, which is then used to label the external dataset.
- 



[^1]: [Generalizing from a Few Examples: A Survey on Few-Shot Learning](https://arxiv.org/pdf/1904.05046.pdf)
