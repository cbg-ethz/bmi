#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*(* Auxiliary function, implementing x Log[x] with the convention that 0 * \[Infinity] = 0. *)*)
(**)
(*XLogX[x_] :=*)
(*    If[x <= 0,0,x Log[x]*)
(*    ];*)
(**)
(*(* The differential entropy h(X) calculated using the PDF function p(x) of an \[DoubleStruckCapitalR]-valued random variable X. *)*)
(**)
(*DifferentialEntropy[distribution_] :=*)
(*    -Integrate[XLogX[PDF[distribution, dummyvariable]], {dummyvariable,*)
(*         -\[Infinity], +\[Infinity]}];*)
(**)
(*(* Calculates the mutual information I(X; Y) where Y = X + N. *)*)
(**)
(*MutualInfoAdditiveNoise[distX_, distN_] :=*)
(*    Module[{x, noise, distY},*)
(*        distY = TransformedDistribution[x + noise, {x \[Distributed] distX, noise *)
(*            \[Distributed] distN}];*)
(*        FullSimplify[DifferentialEntropy[distY] - DifferentialEntropy[*)
(*            distN]]*)
(*    ]*)


(* Uniform X and uniform noise *)
Module[
{
a, b, \[Epsilon],
XDist = UniformDistribution[{a, b}],
NoiseDist = UniformDistribution[{-\[Epsilon], \[Epsilon]}]
},
Assuming[\[Epsilon] > 0 && b > a, MutualInfoAdditiveNoise[XDist, NoiseDist]]
]


(* Gaussian X and Gaussian noise. Note that it can be derived from the correlation *)
Module[
{
	\[Mu]x, \[Sigma]x, \[Mu]n, \[Sigma]n,
	XDist = NormalDistribution[\[Mu]x, \[Sigma]x],
	NoiseDist = NormalDistribution[\[Mu]n, \[Sigma]n]
},
	Assuming[\[Sigma]x > 0 && \[Sigma]n > 0, MutualInfoAdditiveNoise[XDist, NoiseDist]]
]


(* Gaussian X and uniform noise. *)
Module[
{
	\[Mu]x, \[Sigma]x, \[Epsilon],
	XDist = NormalDistribution[\[Mu]x, \[Sigma]x],
	NoiseDist = UniformDistribution[{-\[Epsilon], \[Epsilon]}]
},
	Assuming[\[Sigma]x > 0 && \[Epsilon] > 0, MutualInfoAdditiveNoise[XDist, NoiseDist]]
]


(* Uniform X and Gaussian noise. *)
Module[
{
	a, b, \[Sigma],
	XDist = UniformDistribution[{a, b}],
	NoiseDist = NormalDistribution[0, \[Sigma]]
},
	Assuming[b > a && \[Sigma] > 0, MutualInfoAdditiveNoise[XDist, NoiseDist]]
]



