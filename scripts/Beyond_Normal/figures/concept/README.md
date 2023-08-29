# Concept figure

This is the "cover" figure, being the visual abstract of the project.

We provide it in two versions:
  - `1v1.py`: we start from a bivariate Gaussian distribution 
     and modify them "stairs" functions homeomorphisms.
     We however decided *not to use* this one, as
     (a) perhaps information is lost due to finite precision of floats, and
     (b) the "tails" look quite unnatural, with a lot of mass pushed there.
  - `2v1.py`: we start from a two-dimensional X and one-dimensional Y with jointly
    multinormal distribution. The Y variable is visualised by point color. 
    Then we apply the spiral diffeomorphism to the X variable.
