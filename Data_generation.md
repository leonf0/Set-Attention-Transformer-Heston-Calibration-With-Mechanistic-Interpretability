# Data Generation

## Introduction

Traditional pricing models like Black-Scholes fail to capture the volatility smile that occurs in real option markets. Heston introduces mean reverting stochastic volatility with five parameters $(\kappa, \theta, \xi, \rho, v_0)$ that when combined can explain smile shape, skew and term structure in real option chains.

The problem this project aims to solve is mapping a volatility surface (in the form of a set of $(K, T, \sigma^{\text{IV}})$ triplets), to the Heston parameters $(\kappa, \theta, \xi, \rho, v_0)$. We do this using a machine learning approach, in order to solve this problem in this mannaer we need a labelled dataset where each datapoint is in the form $(\mathbf{X}_i, \Psi_i)$, where $\mathbf{X}_i = \{(\log m_j, \sqrt{\tau_k}, \sigma_j^{\text{IV}})\}$ is the implied volatility surface, and $\Psi_i = (v_0, \kappa, \theta, \xi, \rho)$ is a parameter vector of the calibrated Heston model proper to this surface.

## The Heston Model

The Heston model specifies the dynamics of both an asset price $S_t$ and its instantaneous variance $v_t$, according to the following SDEs:

$$dS_t = r S_t dt + \sqrt{v_t} S_t dW_t^S$$

$$dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v$$

$$d\langle W^S, W^v \rangle_t = \rho  dt$$

Where $v_0 > 0$ is the initial variance of the process, $\theta > 0$ is the long-run variance (ie the value that $v_t$ reverts to), $\kappa > 0$ is the aggressiveness of this mean-reversion, $xi > 0$ is the volatility of the variance process ("vol-of-vol"), and $\rho \in (-1,1)$ is the correlation between the Brownian motions of the asset price and variance (encodes the leverage effect, of higher variance reducing leverage and hence price).

Additonally the model must find some way to guarantee that the variance is strictly positive, this is done via the **Feller Condition.** which enforces that $2\kappa\theta > \xi^2$ which is sufficient to ensure $v_t > 0$ because as $v_t \to 0$, $\xi \sqrt{v_t} \to 0$ and $\kappa(\theta - v_t) \to \kappa\theta$ so the Feller Condition ensures the upward drift is large enough relative to the noise term that $v_t$ cannot reach zero within a finite amount of time.

## Heston Characteristic Function Pricing

