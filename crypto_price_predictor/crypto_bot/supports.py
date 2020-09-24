# Libraries
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# Functions

# To determine the present time (for coinapi information extraction)
def time_present_to_iso_format_utc(minutes_past=15):
    # -15 min want ik wil de coin gegevens van de vorige 15 min
    time_utc = datetime.now() - timedelta(hours=2, minutes=00, seconds=00)
    past_min = time_utc - timedelta(hours=00, minutes=minutes_past, seconds=00)
    past_min_strf = past_min.strftime("%Y-%m-%d %H:%M:%S")

    date = past_min_strf.split()[0]
    hms = past_min_strf.split()[1]

    past_min = datetime(year=int(date.split('-')[0]), month=int(date.split('-')[1]), day=int(date.split('-')[2]),
                        hour=int(hms.split(':')[0]), minute=int(hms.split(':')[1]), second=int(hms.split(':')[2]))
    return past_min.isoformat()


# Return a list of all previous days (the amount is to be chosen)
def dates_by_days_prior(days_to_subtract):
    date_list = []
    for day_number in range(0, days_to_subtract):
        date = datetime.today() - timedelta(days=day_number+1)
        date_list.append(str(date)[:10])
    return date_list


def unix_to_normal_time(x):
    return datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M')

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------