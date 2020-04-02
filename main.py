import click
import numpy as np
from util import DatasetFormatter
from mf import ICFPMF
from interactors import LinearUCB, ThompsonSampling, LinearEGreedy, GLM_UCB


@click.group()
@click.option('-d', default='ml_100k', help=f'Dataset to use [{", ".join(list(DatasetFormatter.BASES_DIRS.keys()))}]')
@click.option('-s', default='users_train_test', help=f'Selection model [{", ".join(list(DatasetFormatter.SELECTION_MODEL.keys()))}]')
@click.option('--load/--save', default=True, help=f'Load(get ready) or save(generate)')
@click.pass_context
def cli(ctx, d, s, load):
    dsf = DatasetFormatter(base=d,selection_model=s)
    if load:
        dsf = dsf.load()
    else:
        dsf.get_base()
        dsf.run_selection_model()
        dsf.save()
    ctx.obj = dsf

@cli.command()
@click.pass_obj
def icfpmf(dsf):
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf.fit(dsf.matrix_users_ratings[dsf.train_uids])

@cli.command()
@click.option('--epsilon', default=0.5, help=f'Epsilon')
@click.pass_obj
def linearegreedy(dsf,epsilon):
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf = mf.load()
    interactor = LinearEGreedy(epsilon=epsilon,
                          var=mf.var,
                          user_lambda=mf.user_lambda,
                          consumption_matrix=dsf.matrix_users_ratings,
                          )
    interactor.interact(dsf.test_uids, mf.items_means)


@cli.command()
@click.pass_obj
def thompsonsampling(dsf):
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf = mf.load()
    interactor = ThompsonSampling(var=mf.var,
                             user_lambda=mf.user_lambda,
                             consumption_matrix=dsf.matrix_users_ratings,
    )
    interactor.interact(dsf.test_uids, mf.items_means, mf.items_covs)

    
@cli.command()
@click.option('--alpha', default=0.5)
@click.pass_obj
def linearucb(dsf,alpha):
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf = mf.load()
    interactor = LinearUCB(alpha=alpha,
                           var=mf.var,
                           user_lambda=mf.user_lambda,
                           consumption_matrix=dsf.matrix_users_ratings)
    interactor.interact(dsf.test_uids, mf.items_means)

@cli.command()
@click.option('-c', default=0.5)
@click.pass_obj
def glmucb(dsf,c):
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf = mf.load()
    interactor = GLM_UCB(c=c,
                           var=mf.var,
                           user_lambda=mf.user_lambda,
                           consumption_matrix=dsf.matrix_users_ratings)
    interactor.interact(dsf.test_uids, mf.items_means)


cli()

# d = DatasetFormatter()
# d.get_base()
# d.run_selection_model()

# observed_ui = np.nonzero(d.matrix_users_ratings) # itens observed by some user
# d.matrix_users_ratings = d.matrix_users_ratings

# model = ICFPMF()
# model.fit(d.matrix_users_ratings# [d.train_uids,:]
# )





# # from sklearn.decomposition import NMF
# from sklearn.decomposition import PCA

# model = PCA(n_components=40)
# W = model.fit_transform(d.matrix_users_ratings)
# H = model.components_
# print(np.sqrt(np.mean((np.dot(W,H)[observed_ui] - d.matrix_users_ratings[observed_ui])**2)))
