import pandas as pd


def filenames_to_datetime_table(filenames, format):
    fn_table = pd.DataFrame({"filenames": filenames})

    if format != "infer":
        datetimes = pd.to_datetime(fn_table["filenames"], format=format, exact=False)
    else:
        datetimes = pd.to_datetime(fn_table["filenames"], infer_datetime_format=True, exact=False)

    fn_table["year"] = datetimes.dt.year
    fn_table["month"] = datetimes.dt.month
    fn_table["day"] = datetimes.dt.day
    fn_table["hour"] = datetimes.dt.hour
    fn_table["minute"] = datetimes.dt.minute

    return fn_table