#!/usr/bin/env python3
import argparse
import html
import numpy as np
import pickle

from lxml import etree


def process_questions(path, workspace, min_score=5):
    qs = dict()
    edges = list()
    cnt = 0
    # Get the root.
    context = iter(etree.iterparse(path, events=("start", "end")))
    event, root = next(context)
    for event, elem in context:
        if not (event == "end" and elem.tag == "row"):
            continue
        cnt += 1
        if cnt % 1e6 == 0:
            print(f"{cnt:,} rows processed")
        if elem.get("PostTypeId") != "1":
            # Row is not a question -> skip.
            continue
        score = int(elem.get("Score", 0))
        if score < min_score:
            # Question has low score -> skip.
            continue
        user_id = elem.get("OwnerUserId")
        if user_id is None:
            # Question has been deleted -> skip.
            continue
        user_id = int(user_id)
        post_id = int(elem.get("Id"))
        tags = elem.get("Tags")
        qs[post_id] = {
            "tags": html.unescape(tags)[1:-1].split("><"),
            "score": score,
        }
        edges.append((user_id, post_id, post_id))
        root.clear()
    with open(workspace, "wb") as f:
        pickle.dump({
            "questions": qs,
            "edges": edges,
        }, f)
    print("{:,} edges".format(len(edges)))


def process_answers(path, workspace, min_score=1):
    with open(workspace, "rb") as f:
        data = pickle.load(f)
    qs = data["questions"]
    edges = data["edges"]
    cnt = 0
    # Get the root.
    context = iter(etree.iterparse(path, events=("start", "end")))
    event, root = next(context)
    for event, elem in context:
        if not (event == "end" and elem.tag == "row"):
            continue
        cnt += 1
        if cnt % 1e6 == 0:
            print(f"{cnt:,} rows processed")
        if elem.get("PostTypeId") != "2":
            # Row is not an answer -> skip.
            continue
        score = int(elem.get("Score", 0))
        if score < min_score:
            # Answer has low score -> skip.
            continue
        parent_id = int(elem.get("ParentId", -1))
        if parent_id not in qs:
            # We didn't keep the associated question -> skip.
            continue
        user_id = elem.get("OwnerUserId")
        if user_id is None:
            # Question has been deleted -> skip.
            continue
        user_id = int(user_id)
        post_id = int(elem.get("Id"))
        edges.append((user_id, parent_id, post_id))
        root.clear()
    with open(workspace, "wb") as f:
        pickle.dump({
            "questions": qs,
            "edges": edges,
        }, f)
    print("{:,} edges".format(len(edges)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=("questions", "answers"))
    parser.add_argument("path")
    parser.add_argument("--workspace", default="workspace.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.which == "questions":
        process_questions(args.path, args.workspace)
    elif args.which == "answers":
        process_answers(args.path, args.workspace)
    else:
        raise ValueError("subcommand does not exist")
