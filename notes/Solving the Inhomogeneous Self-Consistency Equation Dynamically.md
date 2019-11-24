## Solving the Inhomogeneous Self-Consistency Equation Dynamically

Given the equation
$$
\left\langle y_{{\rm y},i}^2 \right\rangle_t = \int_{-\infty}^\infty \tanh^2(x) \mathcal{N}(x,\mu_i,\sigma_i^2) \mathrm{d}x
$$
we approximated $\tanh^2(x)\approx 1-\exp(-x^2)$ , which made the integral
$$
\begin{align}
\left\langle y_{{\rm y},i}^2 \right\rangle_t &= 1 - \frac{1}{\sqrt{2\pi \sigma_i^2}} \int_{-\infty}^\infty \exp\left(-x^2\left[1+\frac{1}{2\sigma_i^2}\right] + x\frac{\mu_i}{\sigma_i^2} - \frac{\mu_i^2}{\sigma_i^4}\right) \mathrm{d}x \\
&= 1-\frac{\exp\left(\frac{\mu_i^2}{4}\frac{1 + 2/\sigma_i^2}{2\sigma_i^2+1}\right)}{\sqrt{1+2\sigma_i^2}} \; . \label{eq:self_consist}
\end{align}
$$
Under the assumption that neither recurrent weights nor node activities are correlated in time or across the population, we have
$$
\begin{align}
\sigma^2_i &= a_i^2 N \left\langle W^2_{ij} \right\rangle_j \mathrm{Var}\left[y_j(t)\right]_{j,t} + \mathrm{Var}\left[E_i(t)\right]_t \\
\mu_i &= a_iN\left\langle W_{ij} \right\rangle_j\left\langle y_j(t)\right\rangle_{j,t} + \left\langle E_i(t) \right\rangle_{t} \; .
\end{align}
$$
The condition that we would like to fulfill is $a_i^2 N \left\langle W^2_{ij}\right\rangle_j = 1 \; \forall i$. Given that we assume balanced weights $\left\langle W_{ij} \right\rangle_j = 0 \; \forall i$, this yields
$$
\begin{align}
\sigma^2_i &= \mathrm{Var}\left[y_j(t)\right]_{j,t} + \mathrm{Var}\left[E_i(t)\right]_t \\
\mu_i &= \left\langle E_i(t) \right\rangle_{t} \; .
\end{align}
$$
If the mean of the input was zero, $\eqref{eq:self_consist}$ would become
$$
\left\langle y^2_i \right\rangle_t = 1-1/\sqrt{1+2\mathrm{Var}\left[y_j(t)\right]_{j,t} + 2 \mathrm{Var}\left[E_i(t)\right]_t}
$$
We are facing the problem that this does not have a general analytical solution. However, we can try to dynamcially solve this equation by simply measuring the right hand side of the equation during adaptation and setting this as a target value for $\left\langle y^2_i \right\rangle_t$ . 

We implemented this using the following update rules:
$$
\begin{align}
X_{{\rm r},i}(t) &= a_i(t-1) \sum_{j=1}^{N_{\rm y}} W_{{\rm r},ij} y_j(t-1) \\
X_{{\rm e},i}(t) &= \sum_{j=1}^{N_{\rm e}} W_{{\rm e},ij} u_j(t) \\
y_i(t) &= \tanh\left(X_{{\rm r},i}(t) + X_{{\rm e},i}(t) - b_i(t-1)\right) \\
\mu_{{\rm y},i}(t) &= (1-\epsilon_\mu)\mu_{{\rm y},i}(t-1) + \epsilon_\mu y_i(t) \\
\mu_{{\rm e},i}(t) &= (1-\epsilon_\mu)\mu_{{\rm e},i}(t-1) + \epsilon_\mu X_{{\rm e},i}(t) \\
\sigma^2_{{\rm y},i}(t) &= (1-\epsilon_\sigma)\sigma^2_{{\rm y},i}(t-1) + \epsilon_\sigma \left[y_i(t) - \mu_{{\rm y},i}(t)\right]^2 \\
\sigma^2_{{\rm e},i}(t) &= (1-\epsilon_\sigma)\sigma^2_{{\rm e},i}(t-1) + \epsilon_\sigma \left[X_{{\rm e},i}(t) - \mu_{{\rm e},i}(t)\right]^2 \\
y^2_{{\rm targ},i}(t) &= 1 - 1/\sqrt{1+2\frac{1}{N_{\rm y}}\sum_{j=1}^{N_{\rm y}}\sigma^2_{{\rm e},j}(t) + 2\sigma^2_{{\rm e},i}(t) } \\
a_i(t) &= a_i(t-1) + \epsilon_{\rm a} \left(y^2_{{\rm targ},i}(t) - y^2_i(t)\right) \\
b_i(t) &= b_i(t-1) + \epsilon_{\rm b} y_i(t)
\end{align}
$$
