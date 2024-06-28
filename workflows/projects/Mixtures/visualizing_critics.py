# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import jax
import jax.numpy as jnp
from subplots_from_axsize import subplots_from_axsize

import bmi
from bmi.samplers import bmm

# -

# ### pick estimators

# +
ksg = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5, 10), n_jobs=4)

nwj = bmi.estimators.neural.NWJEstimator(
    batch_size=512, hidden_layers=(16, 8), standardize=False, test_every_n_steps=250
)
dv = bmi.estimators.neural.DonskerVaradhanEstimator(
    batch_size=512, hidden_layers=(16, 8), standardize=False, test_every_n_steps=250
)
infonce = bmi.estimators.neural.InfoNCEEstimator(
    batch_size=512, hidden_layers=(16, 8), standardize=False, test_every_n_steps=250
)
# -

# ### define distributions

# +
# from tensorflow_probability.substrates import jax as tfp

# arcsinh_bijector = tfp.bijectors.Invert(tfp.bijectors.Sinh())

# +
# dist_student = bmm.MultivariateStudentDistribution(
#     dispersion=0.1 * jnp.eye(2),
#     mean=jnp.array([0., 0.]),
#     dim_x=1, dim_y=1, df=1,
# )

# sampler_student = bmm.FineSampler(dist_student)

# sampler_student_arcsinh = bmm.FineSampler(bmm.transform(
#     dist_student,
#     x_transform=arcsinh_bijector,
#     y_transform=arcsinh_bijector,
# ))

# print(f"MI = {sampler_student.mutual_information():.4f}")

# +
sampler_four_balls = bmm.FineSampler(
    bmm.mixture(
        proportions=jnp.array([0.3, 0.3, 0.2, 0.2]),
        components=[
            bmm.MultivariateNormalDistribution(
                covariance=bmi.samplers.canonical_correlation([0.0]),
                mean=jnp.array([-1.25, -1.25]),
                dim_x=1,
                dim_y=1,
            ),
            bmm.MultivariateNormalDistribution(
                covariance=bmi.samplers.canonical_correlation([0.0]),
                mean=jnp.array([+1.25, +1.25]),
                dim_x=1,
                dim_y=1,
            ),
            bmm.MultivariateNormalDistribution(
                covariance=0.2 * bmi.samplers.canonical_correlation([0.0]),
                mean=jnp.array([-2.5, +2.5]),
                dim_x=1,
                dim_y=1,
            ),
            bmm.MultivariateNormalDistribution(
                covariance=0.2 * bmi.samplers.canonical_correlation([0.0]),
                mean=jnp.array([+2.5, -2.5]),
                dim_x=1,
                dim_y=1,
            ),
        ],
    )
)

print(f"MI = {sampler_four_balls.mutual_information():.4f}")


# -

# ### plot and grid utils


# +
def make_grid(fn, xrange=(-5, 5), yrange=None, steps=101):
    yrange = yrange or xrange
    xs = jnp.linspace(*xrange, steps)
    ys = jnp.linspace(*yrange, steps)
    mxs, mys = jnp.meshgrid(xs, ys, indexing="ij")
    mxs, mys = mxs[..., None], mys[..., None]
    # assumes fn is (n_points, x_dim), (n_points, y_dim) -> (n_points,)
    return jax.vmap(fn)(mxs, mys)


def grid_sampler_pdf(sampler, **kwargs):
    logprob_fn = sampler._dist.dist_joint.log_prob

    def grid_fn(xs, ys):
        xys = jnp.concatenate([xs, ys], axis=-1)
        return jnp.exp(logprob_fn(xys))

    return make_grid(grid_fn, **kwargs)


def grid_sampler_pmi(sampler, **kwargs):
    pmi_fn = sampler._dist.pmi
    return make_grid(pmi_fn, **kwargs)


def grid_critic(neural_estimator, **kwargs):
    critic_fn = jax.vmap(neural_estimator.trained_critic)
    return make_grid(critic_fn, **kwargs)


# -


def plot_grid(ax, grid, xrange=None, yrange=None, steps=None, **kwargs):
    yrange = yrange or xrange
    extent = (*xrange, *yrange) if yrange else None

    ax.imshow(
        grid.T,  # transpose makes the first array dim correspond to the X axis
        origin="lower",
        extent=extent,
        **kwargs,
    )


# +
def remove_dv_dofs(fs):
    assert len(fs.shape) == 2
    return fs - fs.mean()


# axis=1 corresponds to Y, and averaging over Y gives a function c(X)
def remove_nce_dofs(fs):
    assert len(fs.shape) == 2
    return fs - fs.mean(axis=1, keepdims=True)


# -

# ### select sampler

# +
# sampler = sampler_two_balls
# sampler_name = 'two_balls'

sampler = sampler_four_balls
sampler_name = "four_balls"

# sampler = sampler_student
# sampler_name = 'student_dof_1'

# sampler = sampler_student_arcsinh
# sampler_name = 'student_arcsinh_dof_1'

# +
fig, ax = subplots_from_axsize(1, 1, (3, 3), left=0.4, bottom=0.3, top=0.3)

grid_kwargs = dict(xrange=(-5, 5), steps=101)

pdfs = grid_sampler_pdf(sampler, **grid_kwargs)
# pmis = grid_sampler_pmi(sampler, **grid_kwargs)

# ax = axs[0]
ax.set_title("PDF")
plot_grid(ax, pdfs, **grid_kwargs)

# ax = axs[1]
# ax.set_title('PMI')
# plot_grid(ax, pmis, **grid_kwargs)

fig.savefig(f"dist_{sampler_name}.pdf")
# -

# ### train estimators

xs, ys = sampler.sample(5_000, jax.random.PRNGKey(42))

print(f"MI = {sampler._mi:.4f} ± {sampler._mi_stderr:.4f}")

# %%time
ksg.estimate(xs, ys)

nwj.estimate(xs, ys)

dv.estimate(xs, ys)

infonce.estimate(xs, ys)

# ### plot

# +
# prepare grids

pdf = grid_sampler_pdf(sampler, **grid_kwargs)
pmi = grid_sampler_pmi(sampler, **grid_kwargs)
f_nwj = grid_critic(nwj, **grid_kwargs)
f_dv = grid_critic(dv, **grid_kwargs)
f_nce = grid_critic(infonce, **grid_kwargs)

f_dv_mod = remove_dv_dofs(f_dv)
pmi_dv_mod = remove_dv_dofs(pmi)
f_nce_mod = remove_nce_dofs(f_nce)
pmi_nce_mod = remove_nce_dofs(pmi)

# +
# prepare hists

xs_hist, ys_hist = sampler.sample(25_000, jax.random.PRNGKey(43))

pmi_hist = sampler._dist.pmi(xs_hist, ys_hist)
f_nwj_m1_hist = jax.vmap(nwj.trained_critic)(xs_hist, ys_hist) - 1
f_dv_hist = jax.vmap(dv.trained_critic)(xs_hist, ys_hist)

# +
fig, axs = subplots_from_axsize(
    nrows=2,
    ncols=4,
    axsize=([2, 2, 2, 4], 2),
    hspace=0.45,
    wspace=[0.3, 0.3, 1],
    left=0.2,
    top=0.3,
    bottom=0.25,
)

grid_kwargs = dict(xrange=(-5, 5), steps=101)
plot_kwargs = dict(**grid_kwargs, vmin=-10, vmax=5, cmap="jet")
plot_diff_kwargs = dict(**grid_kwargs, vmin=0, vmax=1, cmap="jet")
hist_kwargs = dict(density=True, alpha=0.4, bins=jnp.linspace(-5, 5, 51))


def format_hist_ax(ax):
    ax.set_ylim(0, 1.75)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.legend(loc=(0.0, 0.75), frameon=False, fontsize=11)


# NWJ column
ax = axs[0, 0]
ax.set_title("$f_{NWJ} - 1$")
plot_grid(ax, f_nwj - 1, **plot_kwargs)

ax = axs[1, 0]
ax.set_title("$PMI$")
plot_grid(ax, pmi, **plot_kwargs)


# DV column
ax = axs[0, 1]
ax.set_title("$f_{DV} - mean$")
plot_grid(ax, f_dv_mod, **plot_kwargs)

ax = axs[1, 1]
ax.set_title("$PMI - mean$")
plot_grid(ax, pmi_dv_mod, **plot_kwargs)


# InfoNCE column
ax = axs[0, 2]
ax.set_title("$f_{NCE} - mean_y$")
plot_grid(ax, f_nce_mod, **plot_kwargs)

ax = axs[1, 2]
ax.set_title("$PMI - mean_y$")
plot_grid(ax, pmi_nce_mod, **plot_kwargs)


# hide axes on grid plots
for ax in axs[:, :3].ravel():
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)


# histogram column
ax = axs[0, 3]
ax.hist(pmi_hist, **hist_kwargs, color="green", label="PMI profile")
ax.hist(f_nwj_m1_hist, **hist_kwargs, color="red", label="$f_{NWJ} - 1$ profile")
format_hist_ax(ax)
ax.set_xlim(-3, 2)

ax = axs[1, 3]
ax.hist(pmi_hist - pmi_hist.mean(), **hist_kwargs, color="green", label="PMI profile (shifted)")
ax.hist(
    f_dv_hist - f_dv_hist.mean(), **hist_kwargs, color="red", label="$f_{DV}$ profile (shifted)"
)
format_hist_ax(ax)
ax.set_xlim(-3, 2)


fig.savefig(f"critics_{sampler_name}.pdf")
