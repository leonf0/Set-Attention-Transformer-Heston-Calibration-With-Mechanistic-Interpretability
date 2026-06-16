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

## Heston Characteristic Function Pricing & Fourier Transforms

Under the Heston model,the log asset price ($\log{S_T}$), has no closed-form density (the integrated-variance term is intractable), so we cannot price options using it by integrating a known density against the payoff like the Black–Scholes model does. However although the density is unavailable, its Fourier transform (the characteristic function) is available in closed form

$$\varphi_X(u) = \mathbb{E}\left[e^{iuX}\right] = \int_{-\infty}^{\infty} e^{iux} f_X(x)dx$$

The inversion Theorem says the forward transform doesn't lose any information, thus $f_X(x)$ can be rebuilt from $\varphi_X(u)$ by reassembling its frequency components, and is given by:

$$f_{\log S_T}(x)= \frac{1}{2\pi}\int_{-\infty}^{\infty}e^{-iux}\varphi(u;\tau,\Psi)du$$

Where $\varphi(u;\tau,\Psi)$ is the Heston Characteristic function given by

$$ \varphi(u;\tau,\Psi) = \mathbb{E}^{\mathbb{Q}}\left[e^{iu \log S_T}\right] = e^{iu \log S_0}\varphi_{\mathrm{ret}}(u;\tau,\Psi) $$

$$ \varphi_{\mathrm{ret}}(u;\tau,\Psi) = \mathbb{E}^{\mathbb{Q}}\left[e^{iu \log(S_T/S_0)}\right] $$

## The Closed-Form Characteristic Function & Feyman-Kac Equation

Take a process driven by an SDE with drift $\mu$ and diffusion $\sigma$:

$$ dX_t = \mu(X_t,t)dt + \sigma(X_t,t)dW_t $$

and define a function:

$$f(x,t) = \mathbb{E}\left[g(X_T)\mid X_t = x\right]$$

the expected payoff $g$ at maturity, given you're at state $x$ now. Feynman–Kac states that $f$ solves the PDE:

$$ \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x}+ \frac{1}{2}\sigma^2 \frac{\partial^2 f}{\partial x^2} = 0,\qquad f(x,T) = g(x) $$

Within our context this means that the characteristic function $f(u,\tau)=\mathbb{E}^{\mathbb{Q}}[e^{iu\log S_T}]$ must satisfy the backward PDE which we obtain by applying Feynman–Kac to the two-factor Heston dynamics. 

Because Heston is an *affine* model, meaning the drift and the squared diffusion (the variance) of every state variable are in an affine form (constant-plus-linear), we can look for a solution that is exponential-affine in the state:

$$f = \exp\big(C(\tau) + D(\tau)v_0 + iu\log S_0\big)$$

Substituting this ansatz into the PDE and collecting powers of $v$ collapses the PDE into two ODEs, one being a Riccati equation for $D$ and a direct integral for $C$. 

$$ \frac{dD}{d\tau} = \frac{1}{2}\xi^2 D^2 - \beta D - \frac{1}{2}(u^2 + iu), \qquad D(0)=0 $$

$$ \frac{dC}{d\tau} = (r-q)iu + \kappa\theta D(u,\tau), \qquad C(0)=0 $$

Solving these gives us the closed form below, and removing the spot factor $e^{iu\log S_0}$ leaves the return characteristic function $\varphi_{\mathrm{ret}}=\exp(C+Dv_0)$.

**Intermediate terms.**

$$\beta = \kappa - \rho\xi iu, \qquad d = \sqrt{\beta^{2} + \xi^{2}\left(u^{2}+iu\right)}, \qquad g = \frac{\beta - d}{\beta + d}$$

- $\beta$ represents the effective mean-reversion rate which is the base reversion speed $\kappa$ shifted by the price–variance correlation $\rho\xi iu$ that the frequency-$u$ component feels.
- $d$ is the **discriminant** of the Riccati equation's associated quadratic.
- $g$ is the **integration constant** fixed by the initial condition $D(0)=0$, controlling the shape of the interpolation in $D$.

**Affine coefficients.**

$$D(u,\tau) = \frac{\beta-d}{\xi^{2}}\cdot\frac{1-e^{-d\tau}}{1-ge^{-d\tau}}$$

$D$ is the coefficient on $v_0$ that measures how strongly the distribution depends on current variance. It is zero at $\tau=0$ and saturates to $(\beta-d)/\xi^2$ as $\tau\to\infty$, since mean reversion eventually washes out the initial variance.

$$C(u,\tau) = (r-q)iu\tau + \frac{\kappa\theta}{\xi^{2}}\left[(\beta-d)\tau - 2\log\left(\frac{1-g\,e^{-d\tau}}{1-g}\right)\right]$$

$C$ is the part independent of $v_0$. It accumulates the deterministic drift $(r-q)iu\tau$ and the long-run variance $\theta$ pumped in through the $\kappa\theta$ term over the whole path.

**The characteristic function.**

$$\varphi_{\mathrm{ret}}(u;\tau,\Psi) = \exp\big(C(u,\tau) + D(u,\tau)\,v_0\big)$$

The log of the CF is linear in $v_0$. This affine structure is the reason the closed form exists at all.
