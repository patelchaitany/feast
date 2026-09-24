from datetime import timedelta

import pandas as pd
import pyarrow as pa

from feast.infra.compute_engines.backends.base import DataFrameBackend


class PandasBackend(DataFrameBackend):
    def columns(self, df):
        return df.columns.tolist()

    def from_arrow(self, table):
        return table.to_pandas()

    def join(self, left, right, on, how):
        return left.merge(right, on=on, how=how)

    def groupby_agg(self, df, group_keys, agg_ops):
        return (
            df.groupby(group_keys)
            .agg(
                **{
                    alias: pd.NamedAgg(column=col, aggfunc=func)
                    for alias, (func, col) in agg_ops.items()
                }
            )
            .reset_index()
        )

    def filter(self, df, expr):
        return df.query(expr)

    def to_arrow(self, df):
        return pa.Table.from_pandas(df)

    def to_timedelta_value(self, delta: timedelta):
        return pd.to_timedelta(delta)

    def drop_duplicates(
        self, df, keys, sort_by, ascending: bool = False, keep: int = 1
    ):
        df = df.sort_values(by=sort_by, ascending=ascending)
        if keep == 1:
            return df.drop_duplicates(subset=keys)
        return df.groupby(keys, sort=False, dropna=False).head(keep)

    def rename_columns(self, df, columns: dict[str, str]):
        return df.rename(columns=columns)
