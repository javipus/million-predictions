from aggregation import get_aggregate, neyman_agg, neyman_opt
from utils import get_bdf, score_preds, log_score

bdf = get_bdf()

print("No extremization")
bdf = get_aggregate(bdf, neyman_agg,
                    {"get_d": lambda _: 1,
                     "col_name": "np_no_extr"}
                    )

print("Neyman constant (sqrt 3)...")
bdf = get_aggregate(bdf, neyman_agg,
                    {"get_d": lambda _: 3**(.5),
                     "col_name": "np_lim"}
                    )

fs = .25, .5, .75, 1., 1.5, 2., 5., 10.

for f in fs:
    print(f"{f:.2f} x Neyman...")
    bdf = get_aggregate(bdf, neyman_agg,
                        {"get_d": lambda n: f*neyman_opt(n),
                         "col_name": f"np_f_{f:.2f}"}
                        )

aggs = ['cp', 'mp'] + bdf.columns[-(2+len(fs)):].tolist()

scores = score_preds(bdf, aggs, log_score)
print(scores.describe())
ax = scores.boxplot()
ax.set_yscale('log')
ax.get_figure().show()
