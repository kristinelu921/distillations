[Rethinking Conventional Wisdom in Machine Learning: From Generalization to Scaling](https://arxiv.org/abs/2409.15156)

## Driving Questions

- How do you guide scaling?
- How do you effectively compare models at the scale where only a single experiment is feasible?

## Prereqs
### Neural Tangent Kernels
Pretend network is almost linear in its parameters, $f(x; \theta) \sim f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^T(\theta - \theta_0)$, while the kernel is $$K(x, x') = \nabla_\theta f(x; \theta_0)^T \nabla_\theta f(x'; \theta_0),$$averaged over all hidden parameters. Intuitively, measures how closely two samples $x, x'$ affect the parameters.

Where does it appear? Let $$\mathcal{L}(\theta) = \frac{1}{2} \sum_i ||f_\theta(x_i) - y_i ||^2.$$
Also, $\partial_t f_\theta(x) = -\sum_j \Theta(x, x_j) \nabla_{f_\theta(x_j)} \mathcal{L}$(intuitively, a measure of how similarly to $x_j$ each sample affects parameters scaled by magnitude of change induced by $j$.)

At infinite width, training stays infinitesimally close to the FO Taylor around init due to $\frac{1}{\sqrt{n}}$ normalization. Thus, tangent kernel becomes deterministic as a function of $x$ and $x'$. <- dont fully get this.

This means, can study dynamics at a "function level", and the training becomes a linear ODE to the kernel value. 

## Changes

$$\mathcal{L}(f_\theta; \mathcal{D}) = \underbrace{\left(\mathcal{L}(f_\theta; D) - \mathcal{L}(f_\theta; \mathcal{T})\right)}_{\text{Generalization error}} + \underbrace{\mathcal{L}(f_\theta; \mathcal{T})}_{\text{Approximation error}}$$

When compute > data (before, imgnet era), wanted to increase generalization ability. These are the converged dynamics discoveries:
- Either overparameterize (deep classifiers), (still able to generalize well, why?)
    - Large LR is better because:
        - [Edge of stability](https://arxiv.org/pdf/2103.00065) : top Hessian eigenvalue hovers at $2/\eta$. Increasing $\eta$ lowers maximum curvature compatible with stable local dynamics. So, high LR leads to flatter loss convergents (for full batch GD). 
            - why does it push for sharpness until threshold? bc wants to reduce loss.
            - how applicable is this to full scale training, wiht more complex landscapes? assume quadratic local model, ok because taylor quadratic rep fairly accurate, minus high hessian curvature, higher-order sharp regions, etc.

            - **Intuitively:** model will escape regions of high sharpness naturally, thus converging to areas with low sharpness + high flatness. 
        - [Escape linearization regime](https://arxiv.org/abs/2003.02218): theoretically, high LR would cause infinite-width model to diverge, but work for finite-width models, leading to flatter minima.
            - train on datapoint(x: 1, y: 0), two layer model ($u$, $v$). Define $\lambda$ to be eigenvalue of one sample diff (this is the NTK, one-sample regime, it looks like curvature):$$\lambda_t = \frac{1}{\sqrt{n}}(||u_t||^2 + ||v_t||^2)$$. 
            - $f = \frac{1}{\sqrt{n}}v^Tu, \mathcal{L} = \frac{1}{2} f^2$. GD means $u_{t+1} = u_t - \eta \frac{1}{\sqrt{n}}f_t v_t, \quad v_{t+1} = v_t - \eta \frac{1}{\sqrt{n}}f_t u_t.$
            - That means $f_{t+1} = (1 - \eta \lambda_t + \eta^2 \frac{f_t^2}{n})f_t, \quad \lambda_{t+1} = \lambda_t + \eta \frac{f_t^2}{n} (\eta \lambda_t - 4)$.
            - updates not actually linear when $n$ (width) is small, hessian curvature can actually decrease stably when $\eta < \frac{4}{\lambda}$.
            - **Intuitively**: When LR is high, loss can initially increase while NTK scalar (in 1 dp model) decreases, allowing stabilization later on. "Catapult".
    - Small batch size is better (Keskar):
        - Small-batch SGD has higher noise, no settling in sharp minima. 
        - $B_{crit} \sim \frac{\text{gradient variance}}{||\text{mean gradient}||^2}$.

- Or underparameterize to preserve generalizability, but non-neglible approx error.

When data > compute (now, LLM paradigm), gen error is small, approx error large. This paper tests the transferability of generalization era research for training at scale.

