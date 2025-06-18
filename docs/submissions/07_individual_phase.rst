##############################
7. Individual Phase
##############################

After following the given steps for the first couple of weeks, we were given the opportunity to explore the customizations of machine learning compilers.

**********************************
7.1 Our Pitch
**********************************

7.1.1 Roadmap
====================================

* `Unary Primitives (Tensor Processing Primitives, Table 1) <https://arxiv.org/pdf/2104.05755>`_

    * Square

    * Reciprocal

    * Increment & Decrement

* `Binary Primitives (Tensor Processing Primitives, Table 2) <https://arxiv.org/pdf/2104.05755>`_

    * Add & Sub

    * Mul & Div

    * Max & Min

* Optimizations

    * Optimize new primitives

    * Extend current optimizer (Optional): Dimension Fusion + Dimension Reordering

7.2.1 Risk Evaluation
====================================

* Risks

    * Incorporation of new primitives

    * Considering edge cases (division by zero)

    * Compatibility with our current code - adjustments resulting from this

    * Time management (considering we only have two weeks)

* Rewards / Outcomes

    * Regarding new primitives: enhanced / diversified compiler

    * Regarding optimization: high throughput over the board

**********************************
7.2 Sketch
**********************************

In this phase, we aim to extend our tensor compiler by implementing new primitives and subsequently optimizing their performance. 
Currently, the compiler is rather limited, as supports only a handful of processing primitives. 
To handle more diverse machine learning workloads, we plan to add selected unary and binary primitives, as presented in `Tensor Processing Primitives <https://arxiv.org/pdf/2104.05755>`_.

We will begin by adding a few unary primitives, specifically **Square**, **Reciprocal**, and **Increment & Decrement**.
Since our compiler does not yet support any binary primitives, our next step will be to implement **Add & Sub**, **Mul & Div** and **Max & Min**.

Regarding the implementation, we anticipate that some primitives may be more challenging than others.
For example, we need to account for edge cases, such as division by zero in the **Reciprocal** and **Div** operations.
However, as we have already developed some unary primitives, integrating the new ones into our current framework should be relatively straightfoward.

The situation is slightly different for the binary primitives. 
We do not have a direct starting point for these, but our plan is to integrate them similar to the existing main primitives. 
That said, as this approach is still untested, we may encounter compatibility issues that will need to be addressed along the way.

Importantly, we aim not only to integrate these implementations into our framework but also to achieve a high performance.
Therefore, we plan to optimize these primitives as much as possible. 

Given, that this is a relatively short-term project, we will need to assess our progress as we go.
If time allows, we would also like to further optimize our tensor operations by implementing **dimension fusion** and **dimension reordering**. 
Since we already have some other optimizations in place, integrating these should not result in major issues, although we do expect some challenges along the way.