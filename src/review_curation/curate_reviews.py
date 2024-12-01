from collections import defaultdict
import openreview
import pandas as pd
import pickle

# TMLR uses API V2 client and rest conferences use API V1 client.
v2_client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net') 
v1_client = openreview.Client(baseurl='https://api.openreview.net')


def get_venues():
    venues = v1_client.get_group(id='venues').members
    print(f"Total venues found: {len(venues)}")
    print("Sample: ", venues[0:10])
    return


def curate_iclr_data():
    print("Collecting ICLR data ...")
    all_submissions = defaultdict(dict)
    # Years 2015 and 2016 are unavailable on openreview.net. 
    for year in range(2013, 2018):
        print(year)
        conf_year = f"ICLR_{year}_"
        if year in [2015, 2016]:
            print(f"Skipping ICLR year {year}")
            continue

        notes = openreview.tools.iterget_notes(v1_client, invitation=f'ICLR.cc/{year}/conference/-/submission', details='directReplies')
        submissions_by_forum = {(conf_year + note.forum): note for note in notes}
        all_submissions[year].update(submissions_by_forum)
    
    for year in range(2018, 2024):
        conf_year = f"ICLR_{year}_"
        notes = openreview.tools.iterget_notes(v1_client, invitation=f'ICLR.cc/{str(year)}/Conference/-/Blind_Submission', details='directReplies')
        submissions_by_forum = {(conf_year + note.forum): note for note in notes}
        all_submissions[year].update(submissions_by_forum)

    return all_submissions


def curate_neurips_data():
    print("Collecting NeurIPS data ...")
    all_submissions = defaultdict(dict)
    for year in [2021, 2022]:
        conf_year = f"NeurIPS_{year}_"
        submissions = v1_client.get_all_notes(invitation=f"NeurIPS.cc/{str(year)}/Conference/-/Blind_Submission", details='directReplies')
        for submission in submissions:
            all_submissions[year][conf_year + str(submission.id)] = submission
    return all_submissions


def curate_tmlr_data():
    print("Collecting TMLR data ...")
    all_submissions = defaultdict(dict)
    conf_year = "TMLR_2023_"
    submissions = v2_client.get_all_notes(invitation=f"TMLR/-/Submission", details='directReplies')
    for submission in submissions:
        all_submissions[2023][conf_year + str(submission.id)] = submission
    return all_submissions


def reformat_reviews(all_submissions):
    reviews_df = None
    all_flat_reviews = defaultdict(list)

    for year in all_submissions:
        if year in [2013, 2014]:
            print("We skip ICLR 2013 and 2014 due to no standard format in review and responses.")
            continue
        for subid, submission in all_submissions[year].items():
            reviews = []
            if submission.details:
                if year == 2017:
                    reviews = reviews + [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("/review") or reply["invitation"].endswith("/question")]
                else:
                    if subid.startswith("TMLR"):
                        reviews = reviews + [reply for reply in submission.details["directReplies"] if " ".join(reply["invitations"]).find("/Review") > -1 or " ".join(reply["invitations"]).find("/Decision") > -1]
                    else:
                        reviews = reviews + [reply for reply in submission.details["directReplies"] if reply["invitation"].endswith("Official_Review")]
            else:
                print(submission)

            for rev in reviews:
                flat_rev = {k: rev[k] for k in ['id', 'number', 'forum', 'invitation', 'replyto'] if k in rev}
                if "invitations" in rev:
                    flat_rev["invitation"] = " ".join(rev["invitations"])
                for cont_key in rev['content']:
                    if type(rev['content'][cont_key]) == dict:
                        if 'value' in rev['content'][cont_key]:
                            flat_rev[cont_key] = rev['content'][cont_key]["value"]
                        else:
                            print(f"Inspect dict: {rev}")
                    else:    
                        flat_rev[cont_key] = rev['content'][cont_key]
                flat_rev['year'] = year
                all_flat_reviews[year].append(flat_rev)

        reviews_df = pd.DataFrame.from_dict(all_flat_reviews[year])
        reviews_df.to_pickle(f'./data/reviews/{year}_reviews.pkl')


if __name__ == "__main__":
    all_submissions = curate_iclr_data()
    all_submissions.update(curate_neurips_data())
    all_submissions.update(curate_tmlr_data())

    with open("./data/reviews/submission_reviews.pkl", "wb") as fout:
        pickle.dump(all_submissions, fout)
    
    with open("./data/reviews/submission_reviews.pkl", "rb") as fout:
        all_submissions = pickle.load(fout)

    reformat_reviews(all_submissions)