import click
import numpy as np
from util import DatasetFormatter
from mf import ICFPMF
from interactors import LinearUCB, ThompsonSampling, LinearEGreedy, GLM_UCB

# @click.group()
# def cli2():
#     pass


@click.group(invoke_without_command=True)
@click.option('-d', default='ml_100k', help=f'Dataset to use [{", ".join(list(DatasetFormatter.BASES_DIRS.keys()))}]')
@click.option('-s', default='users_train_test', help=f'Selection model [{", ".join(list(DatasetFormatter.SELECTION_MODEL.keys()))}]')
@click.pass_context
def cli1(ctx, d, s):
    if not(ctx.invoked_subcommand is None):
        if ctx.invoked_subcommand != 'gen-base':
            dsf = DatasetFormatter(base=d,selection_model=s)
            dsf = dsf.load()
            ctx.obj = dsf
        else:
            ctx.ensure_object(dict)
            ctx.obj['d'] = d
            ctx.obj['s'] = s
    else:
        click.echo(ctx.get_help())
@cli1.command()
@click.pass_obj
def gen_base(obj):
    d = obj['d']
    s = obj['s']
    dsf = DatasetFormatter(base=d,selection_model=s)
    dsf.get_base()
    dsf.run_selection_model()
    dsf.save()


@cli1.command()
@click.pass_obj
def icfpmf(dsf):
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf.fit(dsf.matrix_users_ratings[dsf.train_uids])

@cli1.command()
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


@cli1.command()
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

    
@cli1.command()
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

@cli1.command()
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



if __name__ == '__main__':
    cli1()
