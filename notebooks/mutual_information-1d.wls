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
(*        DifferentialEntropy[distY] - DifferentialEntropy[*)
(*            distN]*)
(*    ]*)


(* Uniform X and uniform noise *)
Module[
{
a, b, \[Epsilon],
XDist = UniformDistribution[{0, 1}],
NoiseDist = UniformDistribution[{-\[Epsilon], \[Epsilon]}]
},
Assuming[\[Epsilon] > 0 && b > a, FullSimplify @ MutualInfoAdditiveNoise[XDist, NoiseDist]]
]


(* Gaussian X and Gaussian noise. Note that it can be derived from the correlation *)
Module[
{
	\[Mu]x=1, \[Sigma]x=1, \[Mu]n=2, \[Sigma]n=3/10,
	XDist,
	NoiseDist
},
	XDist = NormalDistribution[\[Mu]x, \[Sigma]x];
	NoiseDist = NormalDistribution[\[Mu]n, \[Sigma]n];
	N[MutualInfoAdditiveNoise[XDist, NoiseDist], 10]
]


(* Gaussian X and uniform noise. *)
Module[
{
	\[Mu]x=1, \[Sigma]x=1, \[Epsilon]=1/10,
	XDist,
	NoiseDist
},
	XDist = NormalDistribution[\[Mu]x, \[Sigma]x];
	NoiseDist = UniformDistribution[{-\[Epsilon], \[Epsilon]}];
	N[MutualInfoAdditiveNoise[XDist, NoiseDist], 3]
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



