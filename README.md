# ML-Course-ProbalisticModel

### UNSW COMP9418 – Assignment 2 2020 T3

### Smart Building Light Control Algorithm

Kan-Lin Lu (z3417618)

1. Introduction

This document provides a smart building light control algorithm specifiation, where the model is built using concept of HMM, built with the given 1-day simulation data.

The simulation is conducted multiple times and the mean of cost and time sits on: 87326.5 cents and 48.1 seconds.

2. Related Background

Hidden Markov Model, HMM, is a form of Dynamic Bayesian Network, where it can be used in modelling transition of unobservable state for a given observation and previous state. An example is shown in Figure 1 a), where Low or High pressure is not observed, but condition of Rain or Dry is. Markov Assumption is also adopted in this case.

3. Exploratory Analysis

We first look at the robot sensors, it is observed that robot sensors are very rare, where out of 2401 single date data point, only less than 1% occurred in a given room. Even though it provides a solid and extreme accurate motion detection (Note there is a difference between motion and count of people, further explained in Method section).

Second, we take a look at the motion sensors reliability, assuming reliable and unreliable referring to same class of sensors. Using the truth table, the following is observed:

| Unreliable\_sensor3 | Count |
| --- | ---|
| Positive | 1295 |
| False Positive | 146 |
| Negative | 894 |
| False Negative | 66 |

| Reliable\_sensor2 | Count |
| --- | ---|
| Positive | 317 |
| False Positive | 76 |
| Negative | 1994 |
| False Negative | 14 |

| Door\_sensor4 | Count |
| --- | ---|
| Positive | 136 |
| False Positive | 0 |
| Negative | 2118 |
| False Negative | 147 |

Reliable Sensor is by around 5-8% more accurate in comparison to unreliable sensor.

On the other hand, door sensor is a special case, as it locates in between 2 areas. The accuracy shows even inferior than unreliable sensor, especially Door Sensor 3 and 1 consist of high false negatives.

![](https://user-images.githubusercontent.com/67504821/106423222-230ede80-64b4-11eb-91b5-df0f78731634.png)

_Figure 1: a) Example of HMM ., b) Illustration of HMM on given Problem, c) Venn Diagram Illustration._

![](https://user-images.githubusercontent.com/67504821/106423348-623d2f80-64b4-11eb-9cd2-1f2c8a81b42b.png)

_Figure 2: Implementation Structure._

We also taken a look at the layout, where r10-11 and r17-r20 do not have enough sensors to rely on, considering how dense the rooms located next to one another. On the other hand, bottom left corner provides a good spread of sensor and rooms, given slight wider area.

We also taken into consideration that people can not travel more than 2 rooms away, except for certain area that is specified.

4. Implementation and Method

Building on the above description, we first establish the concept of predicting whether there are people or not in a given room:

It was first considered in predicting number of people, however, although in theory providing more overall sight, but becomes complicated and can not be directly implemented with HMM.

An illustration of applied HMM concept is shown in Figure 1 b), where P represents People. M and N represents possible emission variables, e.g. Motion Sensor, Neighbouring State and others.

Shown in Figure 1 c), The goal is to find the blue area intersection probability with given evidence, where P(E) represents emission probability table and P(T) represents Transition probability table.

In order to first find transition probability, we read the given ground truth from top to bottom and find the change in state for every 2 rows. However, this results in frequency overfitting, considering how frequent the data is sample at. Therefore, to tackle, Additive Smoothening is applied (value by trial and error), we show in the following r5 (with reliable sensor) as an example, with only 2 extreme alpha ratio, Figure 3 shows an visual representation:

| R5\_t-1 | R5\_t | Pr |
| --- | --- | --- |
| Alpha = 1 |
| True | True | 0.616314 |
| True | False | 0.383686 |
| False | True | 0.0613823 |
| False | False | 0.938618 |
| Alpha = 5000 |
| True | True | 0.255964 |
| True | False | 0.252176 |
| False | True | 0.232317 |
| False | False | 0.314559 |

<img src="https://user-images.githubusercontent.com/67504821/106423433-8b5dc000-64b4-11eb-963d-82459cfacfda.png" width="300">

_Figure 3 : Alpha Value Influence._

Figure 2 illustration the involving overall iteration layers, and the order for iteration is from left to right and top to bottom. Solid and dashed lines represent influence, in other words, we consider whether there is a change in state of connected rooms under the same time frame:

![](https://user-images.githubusercontent.com/67504821/106423498-ab8d7f00-64b4-11eb-9163-7029188ba24a.png)

As an example, State of t for Room 9 takes into account the change in &#39;people&#39; of Room 5 under state of t and t-1.

From this, shows the reason for selecting the right order of iteration is crucial and is carefully considered when constructing Figure 2, the aforementioned reliability of different sensors and layout of the graph.

This further leads to emission probability table, built by children of a given node, see Figure 5 for room 9 as an example.

<img src="https://user-images.githubusercontent.com/67504821/106423734-248cd680-64b5-11eb-85b0-5f14f3c4462d.png" width="150">

_Figure 5: Bayesian Network for Emission._

The Pseudo code for Inference is as follow:

```
For rowintimestamp List:
 for area in iteration List:
  tables = pretrain\_model.load(area)
  Transition = tables[&#39;transition&#39;]
  Emission = tables[&#39;emission&#39;]
  Joint = join(Transition, Emission)
  Q\_evi = {}
  for evi in Evidence:
  Q\_evi[child] = (evi,)
  query(Joint, Q\_evi)
  if true = On, else off
return actions\_dict
```
The complexity of the given model is roughly illustrated in the following, with respect to the above pseudo code:
```
2401 Rows Per date:
 41 Areas per iteration:
  Join requires O(d^n)
  d = number of values for given var
  (Maximum 2^4 + 3^2)
  n = number o variables for given area
  (Maximum 7)
  Maximum 6 Evidence.
```

1. Results

For our final model, we reached 87326.5 cents and 48.1 seconds averaged. We taken a deeper look into the overall model and observed that for Layer 2 and beyond, false prediction of whether there are people or not starts to propagate. Specifically, in layer 7, and high dense room area (r10-r11 and r15-21) with only a single reliable area.

We discussed and considered increasing number of connections between lower layers, however trading off with efficiency, and with our specified restriction, that is no travelling more than 2 rooms away, the given structure provided the most optimal connection.

1. Conclusion and Future Work

In conclusion, the proposed model provides an acceptable outcome. In future, we can further consider establishing connect for same layer nodes, or creating more variables, under emission probability table, in improving performance, e.g. count of people in groups. On the other hand it is suggested in install extra sensor in r7 and r19, which with the given layout will improve overall accuracy.

[1](#sdfootnote1anc) Tutorialandexample: https://www.tutorialandexample.com/hidden-markov-models/
